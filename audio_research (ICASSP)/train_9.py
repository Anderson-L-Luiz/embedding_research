import os
import whisper
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer, BertModel
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Step 1: Load Whisper ASR Model and Transcribe Audio with Timestamps
def transcribe_audio_with_timestamps(file_path, model):
    result = model.transcribe(file_path, word_timestamps=True)
    return result['segments']  # Return the segments containing the text and timestamps

# Initialize Whisper model
whisper_model = whisper.load_model("base")  # You can use "small", "medium", or "large" depending on your needs

# Initialize Wav2Vec2.0 Processor and Model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# Directory containing the AMI Meeting Corpus audio files
audio_directory = "/home/anderson/audio_embedding_research"

# Transcribe all audio files in the directory and process audio
ami_transcripts = []
audio_files = []
for filename in os.listdir(audio_directory):
    if filename.endswith(".wav"):
        file_path = os.path.join(audio_directory, filename)
        transcript_segments = transcribe_audio_with_timestamps(file_path, whisper_model)
        ami_transcripts.append(transcript_segments)
        audio_files.append(file_path)

# Step 2: Split Transcription and Audio into 7-Second Chunks
def group_segments_by_time(segments, chunk_duration=7.0):
    """Groups segments into chunks of approximately 7 seconds each."""
    chunks = []
    current_chunk = ""
    current_chunk_duration = 0.0

    for segment in segments:
        start_time = segment['start']
        end_time = segment['end']
        duration = end_time - start_time
        text = segment['text']

        # If adding this segment exceeds chunk duration, finalize the chunk
        if current_chunk_duration + duration > chunk_duration:
            chunks.append(current_chunk.strip())
            current_chunk = text
            current_chunk_duration = duration
        else:
            current_chunk += " " + text
            current_chunk_duration += duration

    if current_chunk:  # Add any remaining text
        chunks.append(current_chunk.strip())

    return chunks

# Split each transcription into 7-second chunks
all_text_chunks = []
for transcript in ami_transcripts:
    text_chunks = group_segments_by_time(transcript, chunk_duration=7.0)
    all_text_chunks.extend(text_chunks)

print(f"Total text chunks created: {len(all_text_chunks)}")

# Step 3: Generate BERT Embeddings for Each Text Chunk
def get_embedding(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

bert_text_embeddings = [get_embedding(chunk) for chunk in all_text_chunks]
print(f"Computed {len(bert_text_embeddings)} BERT embeddings.")

# Step 4: Extract Audio Segments and Generate Audio Embeddings
def extract_audio_segments(file_path, chunk_duration=7.0, sample_rate=16000):
    waveform, sr = torchaudio.load(file_path)
    resampler = torchaudio.transforms.Resample(sr, sample_rate)
    waveform = resampler(waveform)
    
    num_chunks = int(waveform.size(1) / (chunk_duration * sample_rate))
    audio_chunks = torch.chunk(waveform, num_chunks, dim=1)
    return audio_chunks

# Process each audio file and extract 7-second chunks
audio_embeddings = []
for file_path in audio_files:
    audio_chunks = extract_audio_segments(file_path, chunk_duration=7.0)
    
    for chunk in audio_chunks:
        inputs = processor(chunk.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = wav2vec_model(**inputs)
        audio_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        audio_embeddings.append(audio_embedding)

print(f"Computed {len(audio_embeddings)} audio embeddings.")

# Step 5: Summarize the Transcribed Text with LLaMA 3 8B
huggingface_token="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"
# Load LLaMA 3 8B model and tokenizer (requires Hugging Face access)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_auth_token=huggingface_token)
llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", use_auth_token=huggingface_token)

def summarize_text_llama(text, model, tokenizer, max_new_tokens=100):
    inputs = tokenizer("summarize: " + text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    
    # Generate the summary using max_new_tokens instead of max_length
    outputs = model.generate(
        inputs['input_ids'], 
        attention_mask=inputs['attention_mask'], 
        max_new_tokens=max_new_tokens, 
        num_beams=4, 
        early_stopping=True
    )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# Generate summary for the full transcript using LLaMA
full_transcript = " ".join(all_text_chunks)
ami_summary = summarize_text_llama(full_transcript, llama_model, tokenizer)
print("Generated Summary with LLaMA:", ami_summary)

# Step 6: Get BERT Embedding for the Summary
summary_embedding = get_embedding(ami_summary)

# Step 7: Project Embeddings to a Common Dimensional Space
class ProjectionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionModel, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.projection(x)

# Assuming 768 for Wav2Vec2 and 256 for DNN model input
wave2vec_dim = 768
bert_dim = len(bert_text_embeddings[0])  # Assuming BERT embeddings have same dimension
common_dim = 256  # Project to the common dimensional space

# Create the projection models
projection_model = ProjectionModel(wave2vec_dim, common_dim)
bert_projection_model = ProjectionModel(bert_dim, common_dim)

# Project audio, text, and summary embeddings into the common space
audio_embeddings_projected = [projection_model(torch.tensor(embed, dtype=torch.float32)).detach().numpy() for embed in audio_embeddings]
bert_text_embeddings_projected = [bert_projection_model(torch.tensor(embed, dtype=torch.float32)).detach().numpy() for embed in bert_text_embeddings]
summary_embedding_projected = bert_projection_model(torch.tensor(summary_embedding, dtype=torch.float32)).detach().numpy()

print(f"Projected {len(audio_embeddings_projected)} audio embeddings and {len(bert_text_embeddings_projected)} text embeddings.")

# Step 8: Compare embeddings with summary embedding and calculate significance

def calculate_significance(cosine_similarity_value):
    # Scale cosine similarity [-1, 1] to significance [0, 100]
    significance = ((cosine_similarity_value + 1) / 2) * 100
    return significance

# Compare audio embeddings with projected summary embedding
audio_significance = []
for i, audio_embed in enumerate(audio_embeddings_projected):
    similarity = cosine_similarity(audio_embed.reshape(1, -1), summary_embedding_projected.reshape(1, -1))[0][0]
    significance = calculate_significance(similarity)
    audio_significance.append(significance)
    print(f"Audio Embedding {i+1} Significance: {significance}%")

# Compare text embeddings with projected summary embedding
text_significance = []
for i, text_embed in enumerate(bert_text_embeddings_projected):
    similarity = cosine_similarity(text_embed.reshape(1, -1), summary_embedding_projected.reshape(1, -1))[0][0]
    significance = calculate_significance(similarity)
    text_significance.append(significance)
    print(f"Text Embedding {i+1} Significance: {significance}%")

# Calculate and print mean significance
mean_audio_significance = np.mean(audio_significance)
mean_text_significance = np.mean(text_significance)

print(f"Mean Significance for Audio Embeddings: {mean_audio_significance}%")
print(f"Mean Significance for Text Embeddings: {mean_text_significance}%")

# Step 9: Use Projected Embeddings for Further Analysis or Training
class DNNModel(nn.Module):
    def __init__(self, input_dim):
        super(DNNModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.layer3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.5)
        self.output = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        x = torch.relu(self.layer3(x))
        x = self.dropout3(x)
        x = self.output(x)
        return x

# Learning Rate Finder
from torch.optim.lr_scheduler import LambdaLR

def find_lr(model, criterion, optimizer, data, end_lr=10, num_iter=100):
    lrs = []
    losses = []
    best_loss = float('inf')
    
    def lr_lambda(iteration):
        return end_lr ** (iteration / num_iter)

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    X, y = data
    
    for _ in range(num_iter):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        lrs.append(optimizer.param_groups[0]["lr"])
        losses.append(loss.item())
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        if loss.item() > 4 * best_loss:
            break
    
    return lrs, losses

def plot_lr_vs_loss(lrs, losses):
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.show()

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Training with Early Stopping
def train_model_with_early_stopping(model, criterion, optimizer, data, epochs=200, patience=5, min_delta=0):
    X, y = data
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
        # Early stopping check
        early_stopping(loss.item())
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Plot the training loss over epochs
    plt.plot(range(len(losses)), losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.show()

# Example usage
data = (torch.tensor(audio_embeddings_projected, dtype=torch.float32),
        torch.tensor([[1]] * len(audio_embeddings_projected), dtype=torch.float32))  # Example labels

input_dim = data[0].shape[1]
model = DNNModel(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-7)

lrs, losses = find_lr(model, criterion, optimizer, data)
plot_lr_vs_loss(lrs, losses)

optimal_lr = lrs[losses.index(min(losses))]
print(f"Optimal Learning Rate: {optimal_lr}")

# Train the model with Early Stopping using the optimal learning rate
model = DNNModel(input_dim)
optimizer = optim.Adam(model.parameters(), lr=optimal_lr)

train_model_with_early_stopping(model, criterion, optimizer, data, epochs=200, patience=5, min_delta=1e-4)
