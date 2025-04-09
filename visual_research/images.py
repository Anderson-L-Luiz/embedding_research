import os
import torch
import whisper
from transformers import pipeline
from PIL import Image
import cv2
import numpy as np
from sentence_transformers import SentenceTransformer, util
import json
import matplotlib.pyplot as plt

# ----------------------- Setup -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = SentenceTransformer('clip-ViT-B-32')
llama_pipeline = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-1B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)
whisper_model = whisper.load_model("base")

video_path = "ee_33.mp4"

# ------------------- Transcribe Audio -------------------
whisper_result = whisper_model.transcribe(video_path)
transcript = whisper_result["text"]

# ------------------- Summarize Transcript -------------------
messages = [
    {"role": "system", "content": "You are an assistant that summarizes transcripts."},
    {"role": "user", "content": "Please provide a concise summary of the following transcript:\n\n" + transcript}
]
llama_outputs = llama_pipeline(messages, max_new_tokens=256)

def parse_generated_text(generated_text):
    if isinstance(generated_text, str):
        return generated_text
    if isinstance(generated_text, list):
        return " ".join(str(token.get('text') or token.get('generated_token') or token) for token in generated_text)
    return str(generated_text)

generated_text_field = llama_outputs[0].get("generated_text")
generated_text = parse_generated_text(generated_text_field)
reference_summary = generated_text.split("Please provide a concise summary of the following transcript:")[-1].strip()

# ------------------- Embedding Summary -------------------
text_embedding = clip_model.encode([reference_summary], convert_to_tensor=True, normalize_embeddings=True)

# ------------------- Extract Frames -------------------
def extract_frames(video_path, sample_rate=60):
    cap = cv2.VideoCapture(video_path)
    frames, images = [], []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * sample_rate)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            frames.append(img)
            images.append(frame_rgb)
        frame_count += 1
    cap.release()
    return frames, images

frames, images_np = extract_frames(video_path)

# ------------------- Embedding Frames -------------------
image_embeddings = [
    clip_model.encode(frame, convert_to_tensor=True, normalize_embeddings=True)
    for frame in frames
]
image_embeddings_tensor = torch.stack(image_embeddings)
video_embedding = torch.mean(image_embeddings_tensor, dim=0)
video_embedding = video_embedding / video_embedding.norm()

# ------------------- Cosine Similarity -------------------
cosine_similarity = util.cos_sim(video_embedding, text_embedding)[0][0].item()
frame_similarities = [util.cos_sim(embed, text_embedding)[0][0].item() for embed in image_embeddings]

# ------------------- Prepare Output Folder -------------------
base_dir = "experiments"
os.makedirs(base_dir, exist_ok=True)
existing = [d for d in os.listdir(base_dir) if d.startswith("experiment")]
numbers = [int(d.replace("experiment", "")) for d in existing if d.replace("experiment", "").isdigit()]
next_num = max(numbers + [0]) + 1
experiment_dir = os.path.join(base_dir, f"experiment{next_num}")
os.makedirs(experiment_dir, exist_ok=True)

# ------------------- Save Results -------------------
with open(os.path.join(experiment_dir, "transcript.txt"), "w", encoding="utf-8") as f:
    f.write(transcript)

with open(os.path.join(experiment_dir, "summary.txt"), "w", encoding="utf-8") as f:
    f.write(reference_summary)

np.save(os.path.join(experiment_dir, "clip_reference.npy"), text_embedding.cpu().numpy())
np.save(os.path.join(experiment_dir, "clip_video.npy"), video_embedding.cpu().numpy())

with open(os.path.join(experiment_dir, f"{cosine_similarity:.4f}.txt"), "w") as f:
    pass

# Save similarity plot
plt.figure(figsize=(10, 5))
plt.plot(frame_similarities, marker='o')
plt.title('Frame-to-Summary Similarity')
plt.xlabel('Frame Index')
plt.ylabel('Cosine Similarity')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(experiment_dir, "similarity_plot.png"))
plt.close()

# Save individual frames
frame_dir = os.path.join(experiment_dir, "frames")
os.makedirs(frame_dir, exist_ok=True)
for idx, frame_np in enumerate(images_np):
    Image.fromarray(frame_np).save(os.path.join(frame_dir, f"frame_{idx:03}.jpg"))

print(f"\n✅ Experiment saved in: {experiment_dir}")
