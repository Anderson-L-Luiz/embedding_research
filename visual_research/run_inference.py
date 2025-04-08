import torch
import clip
from PIL import Image
import cv2
import numpy as np

# Set up device and load the CLIP model and its preprocess function.
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# -------------------------------
# Step 1: Encode the Reference Summary
# -------------------------------
# Replace with your LLM-generated summary
reference_summary = (
    "This lecture covers the fundamentals of machine learning, including supervised learning, "
    "unsupervised learning, and reinforcement learning with detailed examples."
)

# Tokenize and generate text embedding
text_tokens = clip.tokenize([reference_summary]).to(device)
with torch.no_grad():
    text_embedding = model.encode_text(text_tokens)
# Normalize the text embedding
text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

# -------------------------------
# Step 2: Extract Frames from Video
# -------------------------------
def extract_frames(video_path, sample_rate=1):
    """
    Extract frames from the video at a given sampling rate (in seconds).

    Args:
        video_path (str): Path to the video file.
        sample_rate (float): Number of seconds between frames to sample.
    Returns:
        list: List of PIL Image frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        raise ValueError("Error opening video file: " + video_path)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * sample_rate)  # sample one frame per 'sample_rate' seconds
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Convert OpenCV's BGR image to RGB and then to a PIL Image.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            frames.append(image)
        frame_count += 1

    cap.release()
    return frames

# Specify your video file path here.
video_path = "path_to_your_video.mp4"  # Update to your video file location
frames = extract_frames(video_path, sample_rate=1)  # Sampling one frame per second

# -------------------------------
# Step 3: Compute Video Embeddings
# -------------------------------
image_embeddings = []
for frame in frames:
    # Preprocess image and add batch dimension.
    img_input = preprocess(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(img_input)
    # Normalize the image embedding
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    image_embeddings.append(image_features.cpu().numpy())

# Convert list of embeddings into a NumPy array (shape: [n_frames, embedding_dim]).
image_embeddings = np.concatenate(image_embeddings, axis=0)

# Aggregate frame embeddings to form a single video-level embedding (mean pooling).
video_embedding = np.mean(image_embeddings, axis=0)
video_embedding = video_embedding / np.linalg.norm(video_embedding)

# -------------------------------
# Step 4: Compute Cosine Similarity
# -------------------------------
# Flatten the text embedding to a vector.
text_embedding_np = text_embedding.cpu().numpy().flatten()

# Compute cosine similarity (since embeddings are normalized, dot product equals cosine similarity).
cosine_similarity = np.dot(video_embedding, text_embedding_np)

print("Cosine Similarity between the reference summary and the video embedding: {:.4f}"
      .format(cosine_similarity))

