import os
import whisper
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer, BertModel
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Set maximum tokens for LLaMA input
LLAMA_MAX_TOKENS = 128000

# Step 1: Create an experiment folder for each run
def create_experiment_folder(base_dir="experiments"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    experiment_num = len(os.listdir(base_dir)) + 1
    experiment_folder = os.path.join(base_dir, f"experiment_{experiment_num}")
    os.makedirs(experiment_folder)
    return experiment_folder

experiment_folder = create_experiment_folder()

# Step 2: Load Whisper ASR Model and Transcribe Audio with Timestamps
def transcribe_audio_with_timestamps(file_path, model):
    result = model.transcribe(file_path, word_timestamps=True)
    return result['segments']

# Initialize Whisper model
whisper_model = whisper.load_model("base")

# Initialize Wav2Vec2 Processor and Model
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

# Step 3: Split Transcription and Audio into 7-Second Chunks
def group_segments_by_time(segments, chunk_duration=7.0):
    chunks = []
    current_chunk = ""
    current_chunk_duration = 0.0
    for segment in segments:
        start_time = segment['start']
        end_time = segment['end']
        duration = end_time - start_time
        text = segment['text']

        if current_chunk_duration + duration > chunk_duration:
            chunks.append(current_chunk.strip())
            current_chunk = text
            current_chunk_duration = duration
        else:
            current_chunk += " " + text
            current_chunk_duration += duration

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Split each transcription into 7-second chunks
all_text_chunks = []
for transcript in ami_transcripts:
    text_chunks = group_segments_by_time(transcript, chunk_duration=7.0)
    all_text_chunks.extend(text_chunks)

print(f"Total text chunks created: {len(all_text_chunks)}")

# Step 4: Save each text chunk as "chunk_n.txt"
for i, chunk in enumerate(all_text_chunks):
    with open(os.path.join(experiment_folder, f"chunk_{i+1}.txt"), "w") as f:
        f.write(chunk)

# Step 5: Generate Summaries for Each Range of Chunks Using LLaMA
huggingface_token = "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"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_auth_token=huggingface_token)
llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", use_auth_token=huggingface_token)

tokenizer.pad_token = tokenizer.eos_token

def summarize_text_llama(text, model, tokenizer, max_new_tokens=2000):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    outputs = model.generate(
        inputs['input_ids'], 
        attention_mask=inputs['attention_mask'], 
        max_new_tokens=max_new_tokens, 
        num_beams=4, 
        early_stopping=True
    )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# Group text chunks into ranges to fit LLaMA's token limit
def chunk_texts_for_summary(text_chunks, max_tokens=LLAMA_MAX_TOKENS):
    grouped_chunks = []
    current_group = ""
    current_tokens = 0

    for chunk in text_chunks:
        num_tokens = len(tokenizer.encode(chunk))
        if current_tokens + num_tokens > max_tokens:
            grouped_chunks.append(current_group.strip())
            current_group = chunk
            current_tokens = num_tokens
        else:
            current_group += " " + chunk
            current_tokens += num_tokens

    if current_group:
        grouped_chunks.append(current_group.strip())

    return grouped_chunks

# Group chunks into text summaries
grouped_texts = chunk_texts_for_summary(all_text_chunks)

# Step 6: Save summaries as "summary_i_j.txt"
chunk_start = 1
for i, grouped_text in enumerate(grouped_texts):
    chunk_end = chunk_start + len(grouped_text.split()) // LLAMA_MAX_TOKENS - 1
    summary = summarize_text_llama(grouped_text, llama_model, tokenizer)
    with open(os.path.join(experiment_folder, f"summary_{chunk_start}_{chunk_end}.txt"), "w") as f:
        f.write(summary)
    chunk_start = chunk_end + 1

print(f"Total summaries created: {len(grouped_texts)}")

# Step 7: Generate BERT Embeddings for Each Text Chunk and Summary
def get_embedding(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Get BERT-based embeddings for the text chunks
bert_text_embeddings = [get_embedding(chunk) for chunk in all_text_chunks]

# Get summary embedding
summary_embedding = get_embedding(" ".join([summarize_text_llama(chunk, llama_model, tokenizer) for chunk in all_text_chunks]))

# Save the embeddings as .npy files
with open(os.path.join(experiment_folder, "bert_text_embeddings.npy"), "wb") as f:
    np.save(f, np.array(bert_text_embeddings))
with open(os.path.join(experiment_folder, "summary_embedding.npy"), "wb") as f:
    np.save(f, summary_embedding)

print("BERT embeddings and summary embedding saved.")

# Step 8: Project Embeddings to a Common Dimensional Space
class ProjectionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionModel, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.projection(x)

wave2vec_dim = 768
bert_dim = len(bert_text_embeddings[0])
common_dim = 256

# Create the projection models
projection_model = ProjectionModel(wave2vec_dim, common_dim)
bert_projection_model = ProjectionModel(bert_dim, common_dim)

# Project audio, text, and summary embeddings into the common space
audio_embeddings_projected = [projection_model(torch.tensor(embed, dtype=torch.float32)).detach().numpy() for embed in audio_embeddings]
bert_text_embeddings_projected = [bert_projection_model(torch.tensor(embed, dtype=torch.float32)).detach().numpy() for embed in bert_text_embeddings]
summary_embedding_projected = bert_projection_model(torch.tensor(summary_embedding, dtype=torch.float32)).detach().numpy()

print(f"Projected {len(audio_embeddings_projected)} audio embeddings and {len(bert_text_embeddings_projected)} text embeddings.")

# Step 9: Compare Embeddings with Summary Embedding and Calculate Significance
def calculate_significance(cosine_similarity_value):
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
