import os
import whisper
import torch
import torchaudio
import transformers
import numpy as np
import parselmouth  # For extracting prosodic features
from parselmouth.praat import call  # For feature extraction
from transformers import AutoTokenizer, BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from TTS.api import TTS  # Import TTS for text-to-speech conversion
from speechbrain.pretrained import SpeakerRecognition  # For x-vectors
import re  # For text cleaning
import pandas as pd  # For saving prosodic features

# Initialize the SpeechBrain x-vector model
spkrec = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="tmp/xvector_model")

# List of WAV file paths
audio_files = [
    "/home/anderson/audio_embedding_research/amicorpus/IB4010/audio/IB4010.Mix-Headset.wav",
    # Add more file paths
]

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

all_text_chunks = []
ami_transcripts = []

# Loop through the list of audio files
for file_path in audio_files:
    transcript_segments = transcribe_audio_with_timestamps(file_path, whisper_model)
    ami_transcripts.append(transcript_segments)

    full_transcript = " ".join([segment['text'] for segment in transcript_segments])
    text_chunks = split_transcript_into_token_chunks(full_transcript, tokenizer, max_tokens=2048)
    all_text_chunks.extend(text_chunks)

print(f"Total token-based chunks created: {len(all_text_chunks)}")

# Step 3: Save each text chunk as "chunk_n.txt"
for i, chunk in enumerate(all_text_chunks):
    with open(os.path.join(experiment_folder, f"chunk_{i+1}.txt"), "w") as f:
        f.write(chunk)

# Step 4: Summarize the Transcribed Text with LLaMA 3.1 8B
model_id = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={
    "torch_dtype": torch.float16,
    "quantization_config": transformers.BitsAndBytesConfig(load_in_4bit=True),
    "low_cpu_mem_usage": True,
})

def summarize_text_llama_with_pipeline(text, pipeline, max_new_tokens=2000):
    messages = [
        {"role": "system", "content": "Summarize the text received, answer only with the details in ONE paragraph. Do not give any other information in your answers."},
        {"role": "user", "content": f"{text}"}
    ]
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipeline(prompt, max_new_tokens=max_new_tokens, eos_token_id=[pipeline.tokenizer.eos_token_id])
    generated_summary = outputs[0]["generated_text"][len(prompt):].strip()
    return generated_summary

grouped_summaries = []
for chunk in all_text_chunks:
    summary = summarize_text_llama_with_pipeline(chunk, pipeline)
    grouped_summaries.append(summary)

# Save summaries to files
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

bert_text_embeddings = [get_embedding(chunk) for chunk in all_text_chunks]
print(f"Computed {len(bert_text_embeddings)} BERT embeddings.")

summary_embedding = get_embedding(" ".join(grouped_summaries))

with open(os.path.join(experiment_folder, "bert_text_embeddings.npy"), "wb") as f:
    np.save(f, np.array(bert_text_embeddings))
with open(os.path.join(experiment_folder, "summary_embedding.npy"), "wb") as f:
    np.save(f, summary_embedding)

print("BERT embeddings and summary embedding saved.")

# Step 6: Extract Audio Segments and Generate Audio Embeddings
def extract_audio_segments_by_chunks(file_path, num_chunks, sample_rate=16000):
    waveform, sr = torchaudio.load(file_path)
    resampler = torchaudio.transforms.Resample(sr, sample_rate)
    waveform = resampler(waveform)
    
    # Divide waveform into 'num_chunks' chunks
    chunk_size = waveform.size(1) // num_chunks
    audio_chunks = torch.chunk(waveform, num_chunks, dim=1)
    return audio_chunks

# Step 6: Extract X-vectors and Prosodic Features for Each Audio Chunk
audio_embeddings = []
prosodic_features_list = []

for file_path in audio_files:
    # Chunk the audio based on the number of text chunks
    audio_chunks = extract_audio_segments_by_chunks(file_path, len(all_text_chunks), sample_rate=16000)
    
    for chunk in audio_chunks:
        # Extract x-vectors for each audio chunk
        xvector = spkrec.encode_batch(chunk, torch.tensor([16000]))
        audio_embeddings.append(xvector.squeeze().numpy())
        
        # Extract prosodic features
        signal_path = os.path.join(experiment_folder, f"audio_chunk_{len(audio_embeddings)}.wav")
        torchaudio.save(signal_path, chunk, 16000)
        snd = parselmouth.Sound(signal_path)
        pitch = snd.to_pitch()
        intensity = snd.to_intensity()
        
        mean_pitch = pitch.mean()
        min_pitch = pitch.get_minimum()
        max_pitch = pitch.get_maximum()
        
        mean_intensity = intensity.values.mean()
        prosodic_data = {
            'chunk': len(audio_embeddings),
            'mean_pitch': mean_pitch,
            'min_pitch': min_pitch,
            'max_pitch': max_pitch,
            'mean_intensity': mean_intensity
        }
        prosodic_features_list.append(prosodic_data)

# Save prosodic features to a CSV file
prosodic_features_df = pd.DataFrame(prosodic_features_list)
prosodic_features_df.to_csv(os.path.join(experiment_folder, "prosodic_features.csv"), index=False)

print(f"Computed {len(audio_embeddings)} audio embeddings with prosodic features.")

# Save the audio embeddings
with open(os.path.join(experiment_folder, "audio_embeddings.npy"), "wb") as f:
    np.save(f, np.array(audio_embeddings))

# Compare embeddings and calculate significance
def calculate_significance(cosine_similarity_value):
    return ((cosine_similarity_value + 1) / 2) * 100

audio_significance = []
for i, audio_embed in enumerate(audio_embeddings):
    if i < len(grouped_summaries):
        similarity = cosine_similarity(audio_embed.reshape(1, -1), bert_text_embeddings[i].reshape(1, -1))[0][0]
        significance = calculate_significance(similarity)
        audio_significance.append(significance)
        print(f"Audio Embedding {i+1} Significance: {significance}%")

print(f"Mean Significance for Audio Embeddings: {np.mean(audio_significance)}%")
