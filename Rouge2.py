import os
import numpy as np
import json
from rouge import Rouge

# Set experiment folder where the data is saved
experiment_folder = "/home/anderson/audio_embedding_research/experiments/experiment_54"
TOLERANCE = 1e-5  # Tolerance for floating-point comparison

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

# Find max, min, and mean significance values
def calculate_statistics(significance_values):
    max_significance = np.max(significance_values)
    min_significance = np.min(significance_values)
    mean_significance = np.mean(significance_values)
    max_index = np.argmax(significance_values)
    min_index = np.argmin(significance_values)
    return max_significance, min_significance, mean_significance, max_index + 1, min_index + 1  # Snippet numbers start from 1

# Concatenate summaries based on significance
def concatenate_summaries(significance_values, summaries, mean_significance):
    higher_summaries = []
    all_summaries = []
    
    for i, summary in enumerate(summaries):
        all_summaries.append(summary)  # All summaries concatenated
        if significance_values[i] >= mean_significance - TOLERANCE:
            print(f"Snippet {i+1} has significance {significance_values[i]} >= {mean_significance} (mean), adding to higher summaries.")
            higher_summaries.append(summary)  # Higher/equal summaries concatenated
        else:
            print(f"Snippet {i+1} has significance {significance_values[i]} < {mean_significance} (mean), NOT adding to higher summaries.")
    
    # If higher_summaries is empty, print a warning
    if not higher_summaries:
        print("No summaries with significance higher than or equal to the mean.")
    
    # Concatenate text
    concatenated_all_summaries = " ".join(all_summaries)
    concatenated_higher_summaries = " ".join(higher_summaries)
    
    return concatenated_all_summaries, concatenated_higher_summaries

# Save the concatenated summaries to files
def save_concatenated_summaries(concatenated_all, concatenated_higher, experiment_folder, summary_type):
    with open(os.path.join(experiment_folder, f"concatenated_all_{summary_type}_summaries.txt"), "w") as f:
        f.write(concatenated_all)
    with open(os.path.join(experiment_folder, f"concatenated_higher_{summary_type}_summaries.txt"), "w") as f:
        f.write(concatenated_higher)

# Apply ROUGE metric
def apply_rouge_metric(reference, summary):
    if not summary.strip():
        raise ValueError("Hypothesis is empty.")  # Ensures ROUGE won't process empty summaries

    rouge = Rouge()
    scores = rouge.get_scores(summary, reference)
    return scores

# Main process
def process_significance_and_summaries(experiment_folder):
    # Step 1: Load significance data
    text_significance, mean_text_significance, audio_significance, mean_audio_significance = load_significance_data(experiment_folder)
    
    # Step 2: Load summaries
    text_summaries = load_summaries(experiment_folder, "summary")
    audio_summaries = load_summaries(experiment_folder, "summary_audio")

    # Step 3: Calculate statistics for text
    max_text_significance, min_text_significance, mean_text_significance, max_text_snippet, min_text_snippet = calculate_statistics(text_significance)
    print(f"Text - Max: {max_text_significance} (Snippet {max_text_snippet}), Min: {min_text_significance} (Snippet {min_text_snippet}), Mean: {mean_text_significance}")

    # Step 4: Concatenate summaries based on text significance
    concatenated_all_text, concatenated_higher_text = concatenate_summaries(text_significance, text_summaries, mean_text_significance)

    # Step 5: Save concatenated text summaries
    save_concatenated_summaries(concatenated_all_text, concatenated_higher_text, experiment_folder, "text")

    # Step 6: Apply ROUGE metric for text
    try:
        rouge_text_scores = apply_rouge_metric(concatenated_all_text, concatenated_higher_text)
        print("ROUGE Scores for Text:")
        print(json.dumps(rouge_text_scores, indent=2))
        with open(os.path.join(experiment_folder, "rouge_text_scores.json"), "w") as f:
            json.dump(rouge_text_scores, f, indent=2)
    except ValueError:
        print("No higher text summaries to analyze with ROUGE.")

    # Step 7: Calculate statistics for audio
    max_audio_significance, min_audio_significance, mean_audio_significance, max_audio_snippet, min_audio_snippet = calculate_statistics(audio_significance)
    print(f"Audio - Max: {max_audio_significance} (Snippet {max_audio_snippet}), Min: {min_audio_significance} (Snippet {min_audio_snippet}), Mean: {mean_audio_significance}")

    # Step 8: Concatenate summaries based on audio significance
    concatenated_all_audio, concatenated_higher_audio = concatenate_summaries(audio_significance, audio_summaries, mean_audio_significance)

    # Step 9: Save concatenated audio summaries
    save_concatenated_summaries(concatenated_all_audio, concatenated_higher_audio, experiment_folder, "audio")

    # Step 10: Apply ROUGE metric for audio
    try:
        rouge_audio_scores = apply_rouge_metric(concatenated_all_audio, concatenated_higher_audio)
        print("ROUGE Scores for Audio:")
        print(json.dumps(rouge_audio_scores, indent=2))
        with open(os.path.join(experiment_folder, "rouge_audio_scores.json"), "w") as f:
            json.dump(rouge_audio_scores, f, indent=2)
    except ValueError:
        print("No higher audio summaries to analyze with ROUGE.")

# Run the main process
if __name__ == "__main__":
    process_significance_and_summaries(experiment_folder)
