import os
import whisper
import torch
import transformers
from transformers import AutoTokenizer, BertTokenizer, BertModel
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import librosa
import parselmouth
from parselmouth.praat import call
from speechbrain.inference import SpeakerRecognition
from TTS.api import TTS  # Import TTS for text-to-speech conversion

# Initialize the x-vector model from SpeechBrain
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

# Initialize Wav2Vec2 Processor and Model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# Step 2: Split Transcription into Token-Based Chunks (2048 tokens)
def split_transcript_into_token_chunks(transcript, tokenizer, max_tokens=2048):
    tokens = tokenizer.tokenize(transcript)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

# Function to clean text for TTS (remove non-Latin characters)
def clean_text_for_tts(text):
    # Retain only Latin characters, numbers, punctuation, and spaces
    return re.sub(r'[^\x00-\x7F]+', '', text)

# Extract x-vector embeddings
def extract_xvector(waveform, sample_rate):
    return spkrec.encode_batch(torch.tensor(waveform).unsqueeze(0), torch.tensor([sample_rate])).squeeze().cpu().numpy()


# Extract biomarker features using Parselmouth and librosa
def extract_biomarker_features(audio_file):
    y, sr = librosa.load(audio_file, sr=16000)
    biomarker_features = {}

    # Load the sound using Parselmouth
    snd = parselmouth.Sound(audio_file)
    pitch = snd.to_pitch()

    # Extract the pitch values from the Pitch object
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values != 0]  # Remove unvoiced parts where pitch is 0

    if len(pitch_values) > 0:
        pitch_mean = np.mean(pitch_values)
        pitch_std = np.std(pitch_values)
    else:
        pitch_mean = 0
        pitch_std = 0

    # Extract formant frequencies using Burg's method
    formants = snd.to_formant_burg()
    f1_mean = np.mean([formants.get_value_at_time(1, t) for t in pitch.ts() if formants.get_value_at_time(1, t) is not None])
    f2_mean = np.mean([formants.get_value_at_time(2, t) for t in pitch.ts() if formants.get_value_at_time(2, t) is not None])

    # Energy calculation
    energy = librosa.feature.rms(y=y)[0].mean()

    # Biomarker features dictionary
    biomarker_features['pitch_mean'] = pitch_mean
    biomarker_features['pitch_std'] = pitch_std
    biomarker_features['f1_mean'] = f1_mean
    biomarker_features['f2_mean'] = f2_mean
    biomarker_features['energy'] = energy

    return biomarker_features

# Save biomarker features to a CSV file
def save_biomarker_features(features, file_path):
    df = pd.DataFrame([features])
    df.to_csv(file_path, index=False)

# Process audio and transcript for each file
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
all_text_chunks = []
ami_transcripts = []

for file_path in audio_files:
    # Transcribe the audio file
    transcript_segments = transcribe_audio_with_timestamps(file_path, whisper_model)
    ami_transcripts.append(transcript_segments)

    # Join the segments into a full transcript
    full_transcript = " ".join([segment['text'] for segment in transcript_segments])

    # Split into token chunks (2048 tokens per chunk)
    text_chunks = split_transcript_into_token_chunks(full_transcript, tokenizer, max_tokens=2048)
    all_text_chunks.extend(text_chunks)

    # Extract x-vectors
    waveform, sr = torchaudio.load(file_path)
    xvector = extract_xvector(waveform.squeeze().numpy(), sr)
    
    # Save x-vectors
    xvector_file = os.path.join(experiment_folder, f"xvector_{os.path.basename(file_path)}.npy")
    np.save(xvector_file, xvector)

    # Extract biomarker features
    biomarker_features = extract_biomarker_features(file_path)

    # Save biomarker features to CSV
    biomarker_file = os.path.join(experiment_folder, f"biomarker_features_{os.path.basename(file_path)}.csv")
    save_biomarker_features(biomarker_features, biomarker_file)

    print(f"Processed {file_path}: x-vectors and biomarkers saved.")

print(f"Total token-based chunks created: {len(all_text_chunks)}")

# Step 3: Save each text chunk as "chunk_n.txt"
for i, chunk in enumerate(all_text_chunks):
    with open(os.path.join(experiment_folder, f"chunk_{i+1}.txt"), "w") as f:
        f.write(chunk)

# Step 4: Summarize the Transcribed Text with LLaMA 3.1 8B
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

    # Generate summary directly from prompt
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

# Step 6: Extract Audio Segments and Generate Audio Embeddings
def extract_audio_segments_by_chunks(file_path, num_chunks, sample_rate=16000):
    waveform, sr = torchaudio.load(file_path)
    resampler = torchaudio.transforms.Resample(sr, sample_rate)
    waveform = resampler(waveform)
    
    chunk_size = waveform.size(1) // num_chunks
    audio_chunks = torch.chunk(waveform, num_chunks, dim=1)
    return audio_chunks

audio_embeddings = []
for file_path in audio_files:
    audio_chunks = extract_audio_segments_by_chunks(file_path, len(all_text_chunks), sample_rate=16000)
    
    for chunk in audio_chunks:  # Process each chunk
        inputs = processor(chunk.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = wav2vec_model(**inputs)
        audio_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        audio_embeddings.append(audio_embedding)

print(f"Computed {len(audio_embeddings)} audio embeddings.")

# Step 7: Text-to-Speech (TTS) Conversion
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")  # Load TTS model

# Convert each summary to audio, resample to 16kHz, and save the files
audio_summary_embeddings = []
for i, summary in enumerate(grouped_summaries):
    # Clean the summary text for TTS
    clean_summary = clean_text_for_tts(summary)
    
    # Path for saving the original audio file from TTS
    audio_file = os.path.join(experiment_folder, f"summary_audio_{i+1}.wav")
    
    if clean_summary.strip():  # Ensure the text is not empty after cleaning
        try:
            tts.tts_to_file(text=clean_summary, file_path=audio_file)  # Convert text summary to audio and save
            print(f"Generated audio summary for Chunk {i+1}.")
        except Exception as e:
            print(f"Error in TTS for Chunk {i+1}: {e}")
    else:
        print(f"Chunk {i+1} contains unsupported characters, skipping TTS.")

    # Load the generated audio file
    waveform, sr = torchaudio.load(audio_file)

    # Resample the audio to 16kHz (Wav2Vec2 model expects 16000 Hz sampling rate)
    target_sample_rate = 16000
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    # Extract audio embeddings using Wav2Vec2
    inputs = processor(waveform.squeeze(), sampling_rate=target_sample_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = wav2vec_model(**inputs)
    
    # Compute the mean embedding for the audio summary
    audio_summary_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    audio_summary_embeddings.append(audio_summary_embedding)

print(f"Computed {len(audio_summary_embeddings)} audio summary embeddings.")

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

# Compare text embeddings with text summary ande
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
