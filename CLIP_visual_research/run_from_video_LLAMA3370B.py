import os
import torch
import whisper
from transformers import pipeline, AutoTokenizer
from PIL import Image
import cv2
import numpy as np
from sentence_transformers import SentenceTransformer, util
import json

# ----------------------------------------------------------------------------------
# Setup Devices and Load Models
# ----------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load SentenceTransformer model for CLIP embeddings.
clip_model = SentenceTransformer('clip-ViT-B-32')

# Setup the LLaMA pipeline for text generation.
llama_model_id = "meta-llama/Llama-3.3-70B-Instruct"
llama_pipeline = pipeline(
    "text-generation",
    model=llama_model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)

# Load the Whisper model (choose size: "base", "small", etc.)
whisper_model = whisper.load_model("base")

# ----------------------------------------------------------------------------------
# Define Video File Path (for both transcription and slide extraction)
# ----------------------------------------------------------------------------------
video_path = "ee_33.mp4"  # UPDATE this with your video file path

# ----------------------------------------------------------------------------------
# Step 1: Transcribe Audio from the Video using Whisper
# ----------------------------------------------------------------------------------
whisper_result = whisper_model.transcribe(video_path)
transcript = whisper_result["text"]
print("Transcript:\n", transcript)

# ----------------------------------------------------------------------------------
# Step 2: Generate a Summary Using LLaMA via a Text Prompt
# ----------------------------------------------------------------------------------
messages = [
    {"role": "system", "content": "You are an assistant that summarizes transcripts."},
    {"role": "user", "content": "Please provide a concise summary of the following transcript:\n\n" + transcript}
]

llama_outputs = llama_pipeline(messages, max_new_tokens=256)

# (Optional) Debug: print the raw LLAMA output structure
# print("LLama Output:\n", json.dumps(llama_outputs[0], indent=2, ensure_ascii=False))

def parse_generated_text(generated_text):
    """
    Helper function to robustly extract generated text.
    It checks if generated_text is a string or a list of dicts
    and tries common keys like 'generated_token' or 'text'.
    """
    if isinstance(generated_text, str):
        return generated_text
    if isinstance(generated_text, list):
        parts = []
        for token in generated_text:
            if isinstance(token, dict):
                # Check for common keys; adjust or add more if needed.
                if 'generated_token' in token:
                    parts.append(token['generated_token'])
                elif 'text' in token:
                    parts.append(token['text'])
                else:
                    parts.append(str(token))
            else:
                parts.append(str(token))
        return " ".join(parts)
    return str(generated_text)

try:
    generated_text_field = llama_outputs[0].get("generated_text")
    generated_text = parse_generated_text(generated_text_field)
except Exception as e:
    raise ValueError("Unexpected format in generated_text: " + str(e))

# Debug: print the parsed generated text if needed.
# print("Parsed Generated Text:\n", generated_text)

# Extract only the summary part by splitting the prompt.
reference_summary = generated_text.split("Please provide a concise summary of the following transcript:")[-1].strip()

print("\nSummary:\n", reference_summary)

# ----------------------------------------------------------------------------------
# Step 3: Compute CLIP Text Embedding for the Reference Summary
# ----------------------------------------------------------------------------------
text_embedding = clip_model.encode(
    [reference_summary], convert_to_tensor=True, normalize_embeddings=True
)

# ----------------------------------------------------------------------------------
# Step 4: Extract Frames (Slides) from the Video
# ----------------------------------------------------------------------------------
def extract_frames(video_path, sample_rate=60):
    """
    Extract frames from the video at a given sampling rate (in seconds).
    
    Args:
        video_path (str): Path to the video file.
        sample_rate (float): Seconds between frames to sample.
    Returns:
        list: List of PIL Image frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        raise ValueError("Error opening video file: " + video_path)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * sample_rate)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            # Convert from BGR (OpenCV) to RGB and then to a PIL Image.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            frames.append(image)
        frame_count += 1

    cap.release()
    return frames

# Extract frames (e.g., one frame every 60 seconds)
frames = extract_frames(video_path, sample_rate=60)

# ----------------------------------------------------------------------------------
# Step 5: Compute CLIP Image Embeddings for the Extracted Frames
# ----------------------------------------------------------------------------------
image_embeddings = []
for frame in frames:
    img_emb = clip_model.encode(frame, convert_to_tensor=True, normalize_embeddings=True)
    image_embeddings.append(img_emb)

if len(image_embeddings) == 0:
    raise ValueError("No valid image embeddings were extracted from the video frames.")

# Aggregate image embeddings via mean pooling.
image_embeddings_tensor = torch.stack(image_embeddings, dim=0)
video_embedding = torch.mean(image_embeddings_tensor, dim=0)
video_embedding = video_embedding / video_embedding.norm()  # Normalize

# ----------------------------------------------------------------------------------
# Step 6: Compute Cosine Similarity Between Text and Video Embeddings
# ----------------------------------------------------------------------------------
cosine_similarity = util.cos_sim(video_embedding, text_embedding)[0][0].item()
print("Cosine Similarity between the reference summary and the video slide embeddings: {:.4f}".format(cosine_similarity))

# ----------------------------------------------------------------------------------
# Step 7: Save the Results to a Text File
# ----------------------------------------------------------------------------------
output_filename = "results.txt"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write("Transcript:\n")
    f.write(transcript + "\n\n")
    f.write("Summary:\n")
    f.write(reference_summary + "\n\n")
    f.write("CLIP Text Embedding (Reference Summary):\n")
    f.write(text_embedding.cpu().numpy().__str__() + "\n\n")
    f.write("CLIP Video Embedding (Aggregated from Frames):\n")
    f.write(video_embedding.cpu().numpy().__str__() + "\n\n")
    f.write("Cosine Similarity:\n")
    f.write(str(cosine_similarity) + "\n")

print(f"\nResults saved to {output_filename}")
