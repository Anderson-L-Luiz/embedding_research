import os
import whisper
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer, BertModel
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Experiment directory creation
def create_experiment_folder(base_dir="experiments"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    experiment_num = len(os.listdir(base_dir)) + 1
    experiment_folder = os.path.join(base_dir, f"experiment_{experiment_num}")
    os.makedirs(experiment_folder)
    return experiment_folder

experiment_folder = create_experiment_folder()

# Step 1: Load Whisper ASR Model and Transcribe Audio with Timestamps
def transcribe_audio_with_timestamps(file_path, model):
    result = model.transcribe(file_path, word_timestamps=True)
    return result['segments']  # Return the segments containing the text and timestamps

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

# Step 2: Split Transcription into Token-Based Chunks (2048 tokens)
def split_transcript_into_token_chunks(transcript, tokenizer, max_tokens=2048):
    tokens = tokenizer.tokenize(transcript)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

# Join transcripts and split them into token-based chunks
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
full_transcript = " ".join([segment['text'] for transcript in ami_transcripts for segment in transcript])
all_text_chunks = split_transcript_into_token_chunks(full_transcript, tokenizer, max_tokens=2048)

print(f"Total token-based chunks created: {len(all_text_chunks)}")

# Step 3: Save each text chunk as "chunk_n.txt"
for i, chunk in enumerate(all_text_chunks):
    with open(os.path.join(experiment_folder, f"chunk_{i+1}.txt"), "w") as f:
        f.write(chunk)

# Step 4: Summarize the Transcribed Text with LLaMA 3 8B using provided structure
model_id = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

# Load the pipeline for text generation
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.float16,
        "quantization_config": transformers.BitsAndBytesConfig(load_in_4bit=True),
        "low_cpu_mem_usage": True,
    },
)

# Summarization function with the updated pipeline structure
def summarize_text_llama_with_pipeline(text, pipeline, max_new_tokens=2000):
    messages = [
        {"role": "system", "content": "Summarize the text received, answer only with the details in ONE paragraph. Do not give any other information in your answers."},
        {"role": "user", "content": f"{text}"}

      #  {"role": "system", "content": "Summarize text, yoonly answer with direct information from the input you receive and you focus on details and not the narrative of the input."},
      #  {"role": "user", "content": f"Summarize the following text, answer only with the summarization text and absolutely nothing else:\n{text}"}
    ]

    # Apply the chat template with prompt
    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # Generate summary
    outputs = pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    # Extract the generated text (summary) from the outputs
    generated_summary = outputs[0]["generated_text"][len(prompt):].strip()
    
    # Print the generated summary for debugging
    print(f"--- GENERATED SUMMARY ---\n{generated_summary}\n")
    
    return generated_summary




#def summarize_text_llama_with_pipeline(text, pipeline, max_new_tokens=2000):
    #prompt = f"Summarize the following text into a brief and coherent summary:\n{text}"

   # terminators = [
  #      pipeline.tokenizer.eos_token_id,
 #       pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
#    ]

    # Generate summary directly from prompt
 #   outputs = pipeline(
       # prompt,
      #  max_new_tokens=max_new_tokens,
     #   eos_token_id=terminators,
    #    do_sample=True,
   #     temperature=0.6,
  #      top_p=0.9,
 #   )

    # Extract the generated text (summary) from the outputs
#    generated_summary = outputs[0]["generated_text"].strip()
    
    # Print the generated summary for debugging
#    print(f"--- GENERATED SUMMARY ---\n{generated_summary}\n")
    
#    return generated_summary


# Summarize each chunk
grouped_summaries = []
for chunk in all_text_chunks:
    summary = summarize_text_llama_with_pipeline(chunk, pipeline)
    grouped_summaries.append(summary)

# Save summaries to files (the actual summary, not the input text)
for i, summary in enumerate(grouped_summaries):
    with open(os.path.join(experiment_folder, f"summary_{i+1}.txt"), "w") as f:
        f.write(summary)

print(f"Total summaries created: {len(grouped_summaries)}")

# Step 5: Generate BERT Embeddings for Each Text Chunk and Summary
def get_embedding(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Generate BERT embeddings for the text chunks
bert_text_embeddings = [get_embedding(chunk) for chunk in all_text_chunks]
print(f"Computed {len(bert_text_embeddings)} BERT embeddings.")

# Get summary embedding
summary_embedding = get_embedding(" ".join(grouped_summaries))

# Save the embeddings
with open(os.path.join(experiment_folder, "bert_text_embeddings.npy"), "wb") as f:
    np.save(f, np.array(bert_text_embeddings))
with open(os.path.join(experiment_folder, "summary_embedding.npy"), "wb") as f:
    np.save(f, summary_embedding)

print("BERT embeddings and summary embedding saved.")

# Step 6: Extract Audio Segments and Generate Audio Embeddings
def extract_audio_segments(file_path, chunk_duration=7.0, sample_rate=16000):
    waveform, sr = torchaudio.load(file_path)
    resampler = torchaudio.transforms.Resample(sr, sample_rate)
    waveform = resampler(waveform)
    
    num_chunks = int(waveform.size(1) / (chunk_duration * sample_rate))
    audio_chunks = torch.chunk(waveform, num_chunks, dim=1)
    return audio_chunks

audio_embeddings = []
for file_path in audio_files:
    audio_chunks = extract_audio_segments(file_path, chunk_duration=7.0)
    
    for chunk in audio_chunks[:5]:  # Limiting to first 5 chunks for the prototype
        inputs = processor(chunk.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = wav2vec_model(**inputs)
        audio_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        audio_embeddings.append(audio_embedding)

print(f"Computed {len(audio_embeddings)} audio embeddings.")

# Step 7: Project Embeddings to a Common Dimensional Space
class ProjectionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionModel, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.projection(x)

# Project audio and text embeddings into a common dimensional space
wave2vec_dim = 768
bert_dim = len(bert_text_embeddings[0])
common_dim = 256

projection_model = ProjectionModel(wave2vec_dim, common_dim)
bert_projection_model = ProjectionModel(bert_dim, common_dim)

audio_embeddings_projected = [projection_model(torch.tensor(embed, dtype=torch.float32)).detach().numpy() for embed in audio_embeddings]
bert_text_embeddings_projected = [bert_projection_model(torch.tensor(embed, dtype=torch.float32)).detach().numpy() for embed in bert_text_embeddings]
summary_embedding_projected = bert_projection_model(torch.tensor(summary_embedding, dtype=torch.float32)).detach().numpy()

# Step 8: Compare Embeddings with Summary Embedding and Calculate Significance
def calculate_significance(cosine_similarity_value):
    significance = ((cosine_similarity_value + 1) / 2) * 100
    return significance

# Compare audio embeddings with summary embedding
audio_significance = []
for i, audio_embed in enumerate(audio_embeddings_projected):
    similarity = cosine_similarity(audio_embed.reshape(1, -1), summary_embedding_projected.reshape(1, -1))[0][0]
    significance = calculate_significance(similarity)
    audio_significance.append(significance)
    print(f"Audio Embedding {i+1} Significance: {significance}%")

# Compare text embeddings with summary embedding
text_significance = []
for i, text_embed in enumerate(bert_text_embeddings_projected):
    similarity = cosine_similarity(text_embed.reshape(1, -1), summary_embedding_projected.reshape(1, -1))[0][0]
    significance = calculate_significance(similarity)
    text_significance.append(significance)
    print(f"Text Embedding {i+1} Significance: {significance}%")

mean_audio_significance = np.mean(audio_significance)
mean_text_significance = np.mean(text_significance)

print(f"Mean Significance for Audio Embeddings: {mean_audio_significance}%")
print(f"Mean Significance for Text Embeddings: {mean_text_significance}%")
