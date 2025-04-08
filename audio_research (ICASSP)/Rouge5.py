import os
import numpy as np
import json
from rouge import Rouge
from pydub import AudioSegment  # Library to handle audio files

# Set experiment folder where the data is saved
experiment_folder = "/home/anderson/audio_embedding_research/experiments/experiment_54"

# Load significance data
def load_significance_data(experiment_folder):
    # Load text and audio significance from .npy files
    text_significance_file = os.path.join(experiment_folder, "text_significance.npy")
    audio_significance_file = os.path.join(experiment_folder, "audio_significance.npy")
    
    text_significance = np.load(text_significance_file)
    audio_significance = np.load(audio_significance_file)

    return text_significance, audio_significance

# Calculate mean significance
def calculate_mean_significance(text_significance, audio_significance):
    combined_significance = np.concatenate([text_significance, audio_significance])
    mean_significance = np.mean(combined_significance)
    return mean_significance

# Load the summaries from text files
def load_summaries(experiment_folder, summary_type):
    summaries = []
    i = 1
    while True:
        summary_file = os.path.join(experiment_folder, f"{summary_type}_{i}.txt")
        if os.path.exists(summary_file):
            with open(summary_file, "r") as f:
                summaries.append(f.read())
            i += 1
        else:
            break
    return summaries

# Load the audio files
def load_audio_summaries(experiment_folder):
    audio_summaries = []
    i = 1
    while True:
        audio_file = os.path.join(experiment_folder, f"summary_audio_{i}.wav")
        if os.path.exists(audio_file):
            audio = AudioSegment.from_wav(audio_file)
            audio_summaries.append(audio)
            i += 1
        else:
            break
    return audio_summaries

# Concatenate significant summaries (both text and audio)
def concatenate_significant_summaries(significance_values, summaries, mean_significance):
    concatenated_summaries = ""
    for significance, summary in zip(significance_values, summaries):
        if significance >= mean_significance:
            concatenated_summaries += summary + "\n"
    return concatenated_summaries

def concatenate_significant_audio(audio_significance, audio_summaries, mean_significance):
    concatenated_audio = AudioSegment.silent(duration=0)  # Start with silence
    for significance, audio in zip(audio_significance, audio_summaries):
        if significance >= mean_significance:
            concatenated_audio += audio  # Concatenate audio
    return concatenated_audio

# Save concatenated summaries
def save_concatenated_summary(concatenated_summary, experiment_folder, summary_type):
    file_path = os.path.join(experiment_folder, f"concatenated_higher_{summary_type}_summaries.txt")
    with open(file_path, "w") as f:
        f.write(concatenated_summary)

def save_concatenated_audio(concatenated_audio, experiment_folder):
    file_path = os.path.join(experiment_folder, "concatenated_higher_audio_summaries.wav")
    concatenated_audio.export(file_path, format="wav")

# Apply ROUGE metric
def apply_rouge_metric(reference, summary):
    rouge = Rouge()
    scores = rouge.get_scores(summary, reference)
    return scores

# Main process
def process_significance_and_summaries(experiment_folder):
    # Step 1: Load significance data
    text_significance, audio_significance = load_significance_data(experiment_folder)
    
    # Step 2: Calculate mean significance
    mean_significance = calculate_mean_significance(text_significance, audio_significance)
    print(f"Mean Significance: {mean_significance}")

    # Step 3: Load summaries (text and audio)
    text_summaries = load_summaries(experiment_folder, "summary")
    audio_summaries = load_audio_summaries(experiment_folder)

    # Step 4: Concatenate significant summaries (text and audio)
    concatenated_text_summary = concatenate_significant_summaries(text_significance, text_summaries, mean_significance)
    concatenated_audio_summary = concatenate_significant_audio(audio_significance, audio_summaries, mean_significance)

    # Step 5: Save concatenated summaries (text and audio)
    save_concatenated_summary(concatenated_text_summary, experiment_folder, "text")
    save_concatenated_audio(concatenated_audio_summary, experiment_folder)

    # Step 6: Apply ROUGE metric for all concatenated summaries
    all_concatenated_text_summary = "\n".join(text_summaries)

    rouge_text_scores = apply_rouge_metric(all_concatenated_text_summary, concatenated_text_summary)

    print("ROUGE Scores for Text Summaries:")
    print(json.dumps(rouge_text_scores, indent=2))

    # Save ROUGE scores to JSON file
    with open(os.path.join(experiment_folder, "rouge_text_scores.json"), "w") as f:
        json.dump(rouge_text_scores, f, indent=2)

# Run the main process
if __name__ == "__main__":
    process_significance_and_summaries(experiment_folder)
