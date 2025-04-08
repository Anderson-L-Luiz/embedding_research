import os
import whisper
import torch
import transformers
from transformers import AutoTokenizer, BertTokenizer, BertModel
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from TTS.api import TTS  # Import TTS for text-to-speech conversion

# List of WAV file paths
audio_files = [
    "/home/anderson/audio_embedding_research/amicorpus/IB4010/audio/IB4010.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/TS3008a/audio/TS3008a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/ES2006a/audio/ES2006a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/IN1013/audio/IN1013.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/TS3010a/audio/TS3010a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/IN1009/audio/IN1009.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/IB4003/audio/IB4003.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/EN2009d/audio/EN2009d.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/ES2015a/audio/ES2015a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/ES2011a/audio/ES2011a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/IS1004a/audio/IS1004a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/ES2010a/audio/ES2010a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/IN1005/audio/IN1005.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/TS3004a/audio/TS3004a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/IN1002/audio/IN1002.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/IS1003a/audio/IS1003a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/IB4011/audio/IB4011.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/EN2002c/audio/EN2002c.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/TS3007a/audio/TS3007a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/ES2009a/audio/ES2009a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/IN1001/audio/IN1001.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/IN1014/audio/IN1014.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/TS3006a/audio/TS3006a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/ES2003a/audio/ES2003a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/IB4001/audio/IB4001.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/IN1016/audio/IN1016.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/IN1007/audio/IN1007.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/TS3003a/audio/TS3003a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/ES2014a/audio/ES2014a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/EN2006a/audio/EN2006a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/ES2016a/audio/ES2016a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/EN2001b/audio/EN2001b.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/ES2007a/audio/ES2007a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/EN2003a/audio/EN2003a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/IB4004/audio/IB4004.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/TS3012a/audio/TS3012a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/IS1008a/audio/IS1008a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/IN1012/audio/IN1012.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/ES2012a/audio/ES2012a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/EN2001e/audio/EN2001e.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/ES2005a/audio/ES2005a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/TS3005a/audio/TS3005a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/EN2001a/audio/EN2001a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/EN2002b/audio/EN2002b.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/IS1009a/audio/IS1009a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/EN2009b/audio/EN2009b.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/IS1000a/audio/IS1000a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/TS3011a/audio/TS3011a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/IS1001a/audio/IS1001a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/EN2001d/audio/EN2001d.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/TS3009a/audio/TS3009a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/ES2013a/audio/ES2013a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/IS1005a/audio/IS1005a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/EN2002d/audio/EN2002d.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/EN2006b/audio/EN2006b.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/IB4002/audio/IB4002.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/EN2002a/audio/EN2002a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/IS1006a/audio/IS1006a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/EN2004a/audio/EN2004a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/EN2005a/audio/EN2005a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/EN2009c/audio/EN2009c.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/IS1007a/audio/IS1007a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/ES2008a/audio/ES2008a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/IB4005/audio/IB4005.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/ES2004a/audio/ES2004a.Mix-Headset.wav",
    "/home/anderson/audio_embedding_research/amicorpus/IN1008/audio/IN1008.Mix-Headset.wav"
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
    # Path for saving the original audio file from TTS
    audio_file = os.path.join(experiment_folder, f"summary_audio_{i+1}.wav")
    tts.tts_to_file(text=summary, file_path=audio_file)  # Convert text summary to audio and save
    print(f"Generated audio summary for Chunk {i+1}.")

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
