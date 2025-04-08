import os
import numpy as np
import json
from rouge import Rouge

# Set experiment folder where the data is saved
experiment_folder = "/home/anderson/audio_embedding_research/experiments/experiment_54"

# Load significance data
def load_significance_data(experiment_folder):
    # Load text and audio significance from .npy files
    text_significance_file = os.path.join(experiment_folder, "text_significance.npy")
    audio_significance_file = os.path.join(experiment_folder, "audio_significance.npy")
    
    text_significance = np.load(text_significance_file)
    audio_significance = np.load(audio_significance_file)

    # Load mean significance from the JSON file
    mean_significance_file = os.path.join(experiment_folder, "mean_significance.json")
    with open(mean_significance_file, "r") as f:
        mean_significance_data = json.load(f)

    mean_text_significance = mean_significance_data["mean_text_significance"]
    mean_audio_significance = mean_significance_data["mean_audio_significance"]
    
    return text_significance, mean_text_significance, audio_significance, mean_audio_significance

# Load the summaries from files
def load_summaries(experiment_folder, summary_type):
    summaries = []
    i = 1
    while True:
        summary_file = os.path.join(experiment_folder, f"{summary_type}_summary_{i}.txt")
        if os.path.exists(summary_file):
            with open(summary_file, "r") as f:
                summaries.append(f.read())
            i += 1
        else:
            break
    return summaries

# Find the highest significance value
def get_highest_significance_summary(significance_values, summaries):
    # Debugging print statements
    print(f"Length of significance values: {len(significance_values)}")
    print(f"Length of summaries: {len(summaries)}")

    if len(significance_values) != len(summaries):
        raise ValueError("Mismatch between significance values and summaries.")
    
    max_index = np.argmax(significance_values)  # Get index of the highest significance value
    highest_summary = summaries[max_index]     # Get the corresponding summary
    return highest_summary, max_index + 1      # Snippet numbers start from 1

# Save the highest significance summaries to files
def save_highest_summary(summary, experiment_folder, summary_type):
    with open(os.path.join(experiment_folder, f"highest_{summary_type}_summary.txt"), "w") as f:
        f.write(summary)

# Apply ROUGE metric
def apply_rouge_metric(reference, summary):
    rouge = Rouge()
    scores = rouge.get_scores(summary, reference)
    return scores

# Main process
def process_significance_and_summaries(experiment_folder):
    # Step 1: Load significance data
    text_significance, _, audio_significance, _ = load_significance_data(experiment_folder)
    
    # Step 2: Load summaries
    text_summaries = load_summaries(experiment_folder, "summary")
    audio_summaries = load_summaries(experiment_folder, "summary_audio")

    # Step 3: Get the highest significance summaries
    highest_text_summary, highest_text_snippet = get_highest_significance_summary(text_significance, text_summaries)
    highest_audio_summary, highest_audio_snippet = get_highest_significance_summary(audio_significance, audio_summaries)
    
    print(f"Highest Text Snippet: {highest_text_snippet}")
    print(f"Highest Audio Snippet: {highest_audio_snippet}")

    # Step 4: Save highest significance summaries
    save_highest_summary(highest_text_summary, experiment_folder, "text")
    save_highest_summary(highest_audio_summary, experiment_folder, "audio")

    # Step 5: Apply ROUGE metric for text
    rouge_text_scores = apply_rouge_metric(highest_text_summary, highest_text_summary)
    print("ROUGE Scores for Text:")
    print(json.dumps(rouge_text_scores, indent=2))
    with open(os.path.join(experiment_folder, "rouge_text_scores.json"), "w") as f:
        json.dump(rouge_text_scores, f, indent=2)

    # Step 6: Apply ROUGE metric for audio
    rouge_audio_scores = apply_rouge_metric(highest_audio_summary, highest_audio_summary)
    print("ROUGE Scores for Audio:")
    print(json.dumps(rouge_audio_scores, indent=2))
    with open(os.path.join(experiment_folder, "rouge_audio_scores.json"), "w") as f:
        json.dump(rouge_audio_scores, f, indent=2)

# Run the main process
if __name__ == "__main__":
    process_significance_and_summaries(experiment_folder)
