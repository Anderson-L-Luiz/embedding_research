import os
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image

# VLM2Vec-specific imports
from src.model import MMEBModel
from src.arguments import ModelArguments
from src.model_utils import load_processor, QWEN2_VL, vlm_image_tokens

def get_best_gpu():
    if not torch.cuda.is_available():
        return 'cpu'

    free_mem = []
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        free_mem.append(torch.cuda.mem_get_info(i)[0])  # free memory

    best_gpu = int(torch.tensor(free_mem).argmax())
    return f"cuda:{best_gpu}"



# ----------------------------------------------------------------------------------
# Setup Devices and Load VLM2Vec Model
# ----------------------------------------------------------------------------------
#device = "cuda" if torch.cuda.is_available() else "cpu"

device = get_best_gpu()

model_args = ModelArguments(
    model_name='Qwen/Qwen2-VL-7B-Instruct',              # Adjust if needed
    checkpoint_path='TIGER-Lab/VLM2Vec-Qwen2VL-7B',      # Adjust if needed
    pooling='last',
    normalize=True,
    model_backbone='qwen2_vl',
    lora=True
)

processor = load_processor(model_args)
model = MMEBModel.load(model_args)
model = model.to(device, dtype=torch.bfloat16)
model.eval()

video_path = "Loki.mp4"  # UPDATE path if needed

reference_summary = "Green, magic, tree, threads, clouds"

# ----------------------------------------------------------------------------------
# Step 3: Compute VLM2Vec Text Embedding
# ----------------------------------------------------------------------------------
# We'll truncate text to 77 tokens to mirror the original approach if needed
inputs = processor(
    text=reference_summary,
    images=None,
    return_tensors="pt",
    truncation=True,
    max_length=77
)
inputs = {key: value.to(device) for key, value in inputs.items()}

with torch.no_grad():
    # Obtain text representations from VLM2Vec
    text_features = model(tgt=inputs)["tgt_reps"]
    # Normalize
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
# Step 5: Compute VLM2Vec Image Embeddings
# ----------------------------------------------------------------------------------
image_embeddings = []
for frame in frames:
    # We need to provide a text prompt that includes the VLM image token to process the image
    # We can give a minimal text prompt (just the image token) or something descriptive
    inputs = processor(
        text=f'{vlm_image_tokens[QWEN2_VL]}',  # Minimal usage of QWEN2_VL token
        images=frame,
        return_tensors="pt"
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # For QWEN2_VL, pixel_values and image_grid_thw normally need an additional dimension
    inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
    inputs['image_grid_thw'] = inputs['image_grid_thw'].unsqueeze(0)

    with torch.no_grad():
        image_features = model(qry=inputs)["qry_reps"]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # shape is [1, D] so we pick [0]
    image_embeddings.append(image_features[0])

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

# Save VLM2Vec model identifiers
model_id_filename = "vlm2vec_model.txt"
with open(os.path.join(experiment_dir, model_id_filename), "w", encoding="utf-8") as f:
    f.write(f"Model Name: {model_args.model_name}\n")
    f.write(f"Checkpoint Path: {model_args.checkpoint_path}\n")

with open(os.path.join(experiment_dir, "vlm2vec_reference.txt"), "w", encoding="utf-8") as f:
    f.write(str(text_features.cpu().float().numpy()))


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

with open(os.path.join(experiment_dir, "vlm2vec_video.txt"), "w", encoding="utf-8") as f:
    f.write(str(video_embedding.cpu().float().numpy()))

print(f"Saved results to: {experiment_dir}")
