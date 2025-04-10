import os
import torch
import whisper
from transformers import pipeline, CLIPProcessor, CLIPModel
from PIL import Image
import cv2
import numpy as np
import torch.nn.functional as F

# ----------------------------------------------------------------------------------
# Setup Devices and Load Models
# ----------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Hugging Face CLIP
clip_model_name = "openai/clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

video_path = "Loki.mp4"  # UPDATE if needed

reference_summary = "Green, magic, tree, threads, clouds"

# ----------------------------------------------------------------------------------
# Step 3: Compute CLIP Text Embedding
# ----------------------------------------------------------------------------------
# Truncate text to 77 tokens to avoid "indices sequence length" errors:
text_inputs = clip_processor(
#    text=[reference_summary],
    text=[reference_summary],
    images=None,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=77
).to(device)

with torch.no_grad():
    # use get_text_features instead of clip_model(...)
    text_features = clip_model.get_text_features(
        input_ids=text_inputs["input_ids"],
        attention_mask=text_inputs["attention_mask"]
    )
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# ----------------------------------------------------------------------------------
# Step 4: Extract Frames from Video
# ----------------------------------------------------------------------------------
def extract_frames(video_file, sample_rate=60):
    cap = cv2.VideoCapture(video_file)
    frames_list = []
    if not cap.isOpened():
        raise ValueError("Error opening video file: " + video_file)
        
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
            frames_list.append(image)
        frame_count += 1

    cap.release()
    return frames_list

frames = extract_frames(video_path, sample_rate=10)

# ----------------------------------------------------------------------------------
# Step 5: Compute CLIP Image Embeddings
# ----------------------------------------------------------------------------------
image_embeddings = []
for frame in frames:
    image_inputs = clip_processor(
        text=None,
        images=frame,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        # use get_image_features instead of clip_model(...)
        image_features = clip_model.get_image_features(
            pixel_values=image_inputs["pixel_values"]
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    image_embeddings.append(image_features[0])  # shape [1, D] => pick [0]

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

with open(os.path.join(experiment_dir, "summary.txt"), "w", encoding="utf-8") as f:
    f.write(reference_summary)

with open(os.path.join(experiment_dir, "clip_reference.txt"), "w", encoding="utf-8") as f:
    # Convert to CPU numpy for storage
    f.write(str(text_features.cpu().numpy()))

# Save CLIP model name
clip_model_filename = clip_model_name.replace("/", "__") + ".txt"
with open(os.path.join(experiment_dir, clip_model_filename), "w", encoding="utf-8") as f:
    f.write(clip_model_name)


# ----------------------------------------------------------------------------------
# Step 7: Save Frames and Cosine Similarities
# ----------------------------------------------------------------------------------
for idx, image_embeds in enumerate(image_embeddings):
    # Save frame
    frame_path = os.path.join(frames_folder, f"frame_{idx}.jpg")
    frames[idx].save(frame_path, "JPEG")

    # Compute cosine similarity with the text embedding (both are [D])
    cosine_similarity_frame = F.cosine_similarity(
        image_embeds.unsqueeze(0), 
        text_features,
        dim=-1
    ).item()

    sim_path = os.path.join(experiment_dir, f"frame{idx}_{cosine_similarity_frame:.4f}.txt")
    with open(sim_path, "w", encoding="utf-8") as f:
        f.write(f"Frame {idx} cosine similarity: {cosine_similarity_frame:.4f}")

# ----------------------------------------------------------------------------------
# Step 8: Save Overall Video Embedding
# ----------------------------------------------------------------------------------
# Stack all image embeddings: shape [num_frames, D]
image_embeddings_tensor = torch.stack(image_embeddings, dim=0)
# Mean across frames
video_embedding = torch.mean(image_embeddings_tensor, dim=0)
# Normalize
video_embedding = video_embedding / video_embedding.norm(dim=-1, keepdim=True)

with open(os.path.join(experiment_dir, "clip_video.txt"), "w", encoding="utf-8") as f:
    f.write(str(video_embedding.cpu().numpy()))

print(f"Saved results to: {experiment_dir}")
