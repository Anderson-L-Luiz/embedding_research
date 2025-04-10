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
clip_model_name = 'clip-ViT-B-32'
clip_model = SentenceTransformer(clip_model_name)

#llama_model_id = "meta-llama/Llama-3.3-70B-Instruct"
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
# Step 2: Generate Short Summary Using LLaMA
# ----------------------------------------------------------------------------------
summary_prompt = f"""
You are an assistant that summarizes transcripts. 
Provide a concise summary (no more than ~100 words) of the main technical points from the transcript below. 
Do NOT repeat the entire transcript verbatim; only give the key concepts.

Transcript:
{transcript}

Summary:
"""

llama_outputs = llama_pipeline(summary_prompt, max_new_tokens=256, temperature=0.7)

# Extract the generated text
generated_text = llama_outputs[0]["generated_text"]

# Optional: parse out anything that appears after "Summary:" to keep it clean
if "Summary:" in generated_text:
    reference_summary = generated_text.split("Summary:")[-1].strip()
else:
    reference_summary = generated_text.strip()

print(reference_summary)

# ----------------------------------------------------------------------------------
# Step 3: Compute CLIP Text Embedding
# ----------------------------------------------------------------------------------
text_embedding = clip_model.encode([reference_summary], convert_to_tensor=True, normalize_embeddings=True)

# ----------------------------------------------------------------------------------
# Step 4: Extract Frames from Video
# ----------------------------------------------------------------------------------
def extract_frames(video_path, sample_rate=60):
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

frames = extract_frames(video_path, sample_rate=10)

# ----------------------------------------------------------------------------------
# Step 5: Compute CLIP Image Embeddings
# ----------------------------------------------------------------------------------
image_embeddings = [
    clip_model.encode(frame, convert_to_tensor=True, normalize_embeddings=True)
    for frame in frames
]

if not image_embeddings:
    raise ValueError("No valid image embeddings extracted.")

# ----------------------------------------------------------------------------------
# Step 6: Create Experiment Directory and Save Metadata
# ----------------------------------------------------------------------------------
base_dir = "experiments"
os.makedirs(base_dir, exist_ok=True)

existing = [d for d in os.listdir(base_dir) if d.startswith("experiment")]
numbers = [int(d.replace("experiment", "")) for d in existing if d.replace("experiment", "").isdigit()]
next_num = max(numbers + [0]) + 1
experiment_dir = os.path.join(base_dir, f"experiment{next_num}")
os.makedirs(experiment_dir, exist_ok=True)

# Create frames folder
frames_folder = os.path.join(experiment_dir, "frames")
os.makedirs(frames_folder, exist_ok=True)

# Save transcript and summary
with open(os.path.join(experiment_dir, "transcript.txt"), "w", encoding="utf-8") as f:
    f.write(transcript)

with open(os.path.join(experiment_dir, "summary.txt"), "w", encoding="utf-8") as f:
    f.write(reference_summary)

with open(os.path.join(experiment_dir, "clip_reference.txt"), "w", encoding="utf-8") as f:
    f.write(str(text_embedding.cpu().numpy()))

# Save CLIP model name
with open(os.path.join(experiment_dir, f"{clip_model_name}.txt"), "w", encoding="utf-8") as f:
    f.write(clip_model_name)

# Save LLaMA model name
llama_model_name_file = llama_model_id.split("/")[-1] + ".txt"
with open(os.path.join(experiment_dir, llama_model_name_file), "w", encoding="utf-8") as f:
    f.write(llama_model_id)

# ----------------------------------------------------------------------------------
# Step 7: Save Frames and Cosine Similarities
# ----------------------------------------------------------------------------------
for idx, (frame, image_embedding) in enumerate(zip(frames, image_embeddings)):
    # Save frame
    frame_path = os.path.join(frames_folder, f"frame_{idx}.jpg")
    frame.save(frame_path, "JPEG")

    # Compute cosine similarity
    cosine_similarity_frame = util.cos_sim(image_embedding, text_embedding)[0][0].item()
    sim_path = os.path.join(experiment_dir, f"frame{idx}_{cosine_similarity_frame:.4f}.txt")
    with open(sim_path, "w", encoding="utf-8") as f:
        f.write(f"Frame {idx} cosine similarity: {cosine_similarity_frame:.4f}")

# ----------------------------------------------------------------------------------
# Step 8: Save Overall Video Embedding
# ----------------------------------------------------------------------------------
image_embeddings_tensor = torch.stack(image_embeddings, dim=0)
video_embedding = torch.mean(image_embeddings_tensor, dim=0)
video_embedding = video_embedding / video_embedding.norm()

with open(os.path.join(experiment_dir, "clip_video.txt"), "w", encoding="utf-8") as f:
    f.write(str(video_embedding.cpu().numpy()))

print(f"Saved results to: {experiment_dir}")
