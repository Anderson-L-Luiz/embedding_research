import os
import whisper
import torch
import torchaudio
import transformers
import numpy as np
from transformers import AutoTokenizer, BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from TTS.api import TTS  # Import TTS for text-to-speech conversion
from speechbrain.pretrained import SpeakerRecognition  # For x-vectors
import librosa
import parselmouth
import pandas as pd

# Initialize the SpeechBrain x-vector model
spkrec = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="tmp/xvector_model")

# List of WAV file paths
audio_files = [
    "/home/anderson/audio_embedding_research/amicorpus/IB4010/audio/IB4010.Mix-Headset.wav"
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
    # Transcribe the audio file
    transcript_segments = transcribe_audio_with_timestamps(file_path, whisper_model)
    ami_transcripts.append(transcript_segments)

    # Join the segments into a full transcript
    full_transcript = " ".join([segment['text'] for segment in transcript_segments])

    # Split into token chunks (2048 tokens per chunk)
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
    print(f"--- GENERATED SUMMARY ---\n{generated_summary}\n")
    return generated_summary

# Summarize each chunk
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

# Step 6: Extract X-vectors (Speaker Embeddings) from Audio
def extract_xvector(audio_file):
    signal, fs = torchaudio.load(audio_file)
    xvector = spkrec.encode_batch(signal, torch.tensor([fs]))
    return xvector.squeeze().numpy()

audio_embeddings = []
for file_path in audio_files:
    xvector = extract_xvector(file_path)
    audio_embeddings.append(xvector)

print(f"Computed {len(audio_embeddings)} audio (x-vector) embeddings.")

# Step 7: Text-to-Speech (TTS) Conversion
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")  # Load TTS model

# Convert each summary to audio, resample to 16kHz, and save the files
audio_summary_embeddings = []
for i, summary in enumerate(grouped_summaries):
    # Path for saving the original audio file from TTS
    audio_file = os.path.join(experiment_folder, f"summary_audio_{i+1}.wav")
    tts.tts_to_file(text=summary, file_path=audio_file)  # Convert text summary to audio and save

    # Extract x-vectors from the generated audio summary
    xvector_summary = extract_xvector(audio_file)
    audio_summary_embeddings.append(xvector_summary)

print(f"Computed {len(audio_summary_embeddings)} audio summary (x-vector) embeddings.")

# Ensure the number of audio embeddings matches the number of audio summary embeddings
if len(audio_embeddings) != len(audio_summary_embeddings):
    print(f"Warning: Mismatch in the number of audio embeddings ({len(audio_embeddings)}) and audio summary embeddings ({len(audio_summary_embeddings)}).")

# Save the audio summary embeddings
with open(os.path.join(experiment_folder, "audio_summary_embeddings.npy"), "wb") as f:
    np.save(f, np.array(audio_summary_embeddings))

# Step 8: Compare Embeddings and Calculate Significance
def calculate_significance(cosine_similarity_value):
    significance = ((cosine_similarity_value + 1) / 2) * 100
    return significance

# Compare audio embeddings with audio summary embeddings
audio_significance = []
for i, audio_embed in enumerate(audio_embeddings):
    if i < len(audio_summary_embeddings):  # Ensure the index doesn't go out of range
        similarity = cosine_similarity(audio_embed.reshape(1, -1), audio_summary_embeddings[i].reshape(1, -1))[0][0]
        significance = calculate_significance(similarity)
        audio_significance.append(significance)
        print(f"Audio Embedding {i+1} Significance with Audio Summary: {significance}%")
    else:
        print(f"Skipping audio embedding {i+1} due to mismatch.")

# Compare text embeddings with text summary embedding
text_significance = []
for i, text_embed in enumerate(bert_text_embeddings):
    similarity = cosine_similarity(text_embed.reshape(1, -1), summary_embedding.reshape(1, -1))[0][0]
    significance = calculate_significance(similarity)
    text_significance.append(significance)
    print(f"Text Embedding {i+1} Significance with Text Summary: {significance}%")

# Calculate mean significance
mean_audio_significance = np.mean(audio_significance)
mean_text_significance = np.mean(text_significance)

print(f"Mean Significance for Audio Embeddings: {mean_audio_significance}%")
print(f"Mean Significance for Text Embeddings: {mean_text_significance}%")
