import os
import torch
import whisper
from transformers import pipeline
from PIL import Image
import cv2
import numpy as np
from sentence_transformers import SentenceTransformer, util
import json

# ----------------------------------------------------------------------------------
# Setup Devices and Load Models
# ----------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = SentenceTransformer('clip-ViT-B-32')

llama_model_id = "meta-llama/Llama-3.2-1B-Instruct"
llama_pipeline = pipeline(
    "text-generation",
    model=llama_model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)

whisper_model = whisper.load_model("base")

# ----------------------------------------------------------------------------------
# Define Video File Path
# ----------------------------------------------------------------------------------
video_path = "ee_33.mp4"  # UPDATE if needed

# ----------------------------------------------------------------------------------
# Step 1: Transcribe Audio
# ----------------------------------------------------------------------------------
whisper_result = whisper_model.transcribe(video_path)
transcript = whisper_result["text"]

# ----------------------------------------------------------------------------------
# Step 2: Generate Summary Using LLaMA
# ----------------------------------------------------------------------------------
messages = [
    {"role": "system", "content": "You are an assistant that summarizes transcripts."},
    {"role": "user", "content": "Please provide a concise summary of the following transcript:\n\n" + transcript}
]

llama_outputs = llama_pipeline(messages, max_new_tokens=256)

def parse_generated_text(generated_text):
    if isinstance(generated_text, str):
        return generated_text
    if isinstance(generated_text, list):
        parts = []
        for token in generated_text:
            if isinstance(token, dict):
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

reference_summary = generated_text.split("Please provide a concise summary of the following transcript:")[-1].strip()

# ----------------------------------------------------------------------------------
# Step 3: Compute CLIP Text Embedding
# ----------------------------------------------------------------------------------
text_embedding = clip_model.encode([reference_summary], convert_to_tensor=True, normalize_embeddings=True)

# ----------------------------------------------------------------------------------
# Step 4: Extract Frames from Video
# ----------------------------------------------------------------------------------
def extract_frames(video_path, sample_rate=10):
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
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            frames.append(image)
        frame_count += 1

    cap.release()
    return frames

frames = extract_frames(video_path, sample_rate=60)

# ----------------------------------------------------------------------------------
# Step 5: Compute CLIP Image Embeddings
# ----------------------------------------------------------------------------------
image_embeddings = [
    clip_model.encode(frame, convert_to_tensor=True, normalize_embeddings=True)
    for frame in frames
]

if not image_embeddings:
    raise ValueError("No valid image embeddings extracted.")

image_embeddings_tensor = torch.stack(image_embeddings, dim=0)
video_embedding = torch.mean(image_embeddings_tensor, dim=0)
video_embedding = video_embedding / video_embedding.norm()

# ----------------------------------------------------------------------------------
# Step 6: Cosine Similarity
# ----------------------------------------------------------------------------------
cosine_similarity = util.cos_sim(video_embedding, text_embedding)[0][0].item()

# ----------------------------------------------------------------------------------
# Step 7: Save Results to experiments/experimentX/
# ----------------------------------------------------------------------------------
base_dir = "experiments"
os.makedirs(base_dir, exist_ok=True)

existing = [d for d in os.listdir(base_dir) if d.startswith("experiment")]
numbers = [int(d.replace("experiment", "")) for d in existing if d.replace("experiment", "").isdigit()]
next_num = max(numbers + [0]) + 1
experiment_dir = os.path.join(base_dir, f"experiment{next_num}")
os.makedirs(experiment_dir, exist_ok=True)

# Save files
with open(os.path.join(experiment_dir, "transcript.txt"), "w", encoding="utf-8") as f:
    f.write(transcript)

with open(os.path.join(experiment_dir, "summary.txt"), "w", encoding="utf-8") as f:
    f.write(reference_summary)

with open(os.path.join(experiment_dir, "clip_reference.txt"), "w", encoding="utf-8") as f:
    f.write(str(text_embedding.cpu().numpy()))

with open(os.path.join(experiment_dir, "clip_video.txt"), "w", encoding="utf-8") as f:
    f.write(str(video_embedding.cpu().numpy()))

similarity_filename = os.path.join(experiment_dir, f"{cosine_similarity:.4f}.txt")
open(similarity_filename, "w").close()

print(f"Saved results to: {experiment_dir}")
