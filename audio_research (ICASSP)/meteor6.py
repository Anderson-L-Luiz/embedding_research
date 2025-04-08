import os
import numpy as np
import json
from pydub import AudioSegment  # Library to handle audio files
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

import nltk
nltk.data.path.append('/home/anderson/nltk_data/')

# Ensure that the necessary NLTK resources are downloaded
nltk.download('punkt')

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

# Calculate mean, min, max significance, and snippet index
def calculate_significance_stats(significance_values):
    mean_significance = np.mean(significance_values)
    min_significance = np.min(significance_values)
    max_significance = np.max(significance_values)
    min_snippet = np.argmin(significance_values) + 1  # +1 to match snippet numbering
    max_snippet = np.argmax(significance_values) + 1  # +1 to match snippet numbering
    return mean_significance, min_significance, max_significance, min_snippet, max_snippet

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
    num_snippets_used = 0
    for significance, summary in zip(significance_values, summaries):
        if significance >= mean_significance:
            concatenated_summaries += summary + "\n"
            num_snippets_used += 1
    return concatenated_summaries, num_snippets_used

def concatenate_significant_audio(audio_significance, audio_summaries, mean_significance):
    concatenated_audio = AudioSegment.silent(duration=0)  # Start with silence
    num_snippets_used = 0
    for significance, audio in zip(audio_significance, audio_summaries):
        if significance >= mean_significance:
            concatenated_audio += audio  # Concatenate audio
            num_snippets_used += 1
    return concatenated_audio, num_snippets_used

# Save concatenated summaries
def save_concatenated_summary(concatenated_summary, experiment_folder, summary_type):
    file_path = os.path.join(experiment_folder, f"concatenated_higher_{summary_type}_summaries.txt")
    with open(file_path, "w") as f:
        f.write(concatenated_summary)

def save_concatenated_audio(concatenated_audio, experiment_folder):
    file_path = os.path.join(experiment_folder, "concatenated_higher_audio_summaries.wav")
    concatenated_audio.export(file_path, format="wav")

# Apply METEOR metric with tokenization
def apply_meteor_metric(reference, summary):
    # Tokenize both reference and summary
    reference_tokens = word_tokenize(reference)
    summary_tokens = word_tokenize(summary)
    
    # Compute METEOR score using tokenized inputs
    score = meteor_score([reference_tokens], summary_tokens)
    return score

# Main process
def process_significance_and_summaries(experiment_folder):
    # Step 1: Load significance data
    text_significance, audio_significance = load_significance_data(experiment_folder)
    
    # Step 2: Calculate significance stats for text and audio
    text_mean, text_min, text_max, text_min_snippet, text_max_snippet = calculate_significance_stats(text_significance)
    audio_mean, audio_min, audio_max, audio_min_snippet, audio_max_snippet = calculate_significance_stats(audio_significance)

    # Step 3: Print significance stats for text and audio
    print(f"Text - Max: {text_max} (Snippet {text_max_snippet}), Min: {text_min} (Snippet {text_min_snippet}), Mean: {text_mean}")
    print(f"Audio - Max: {audio_max} (Snippet {audio_max_snippet}), Min: {audio_min} (Snippet {audio_min_snippet}), Mean: {audio_mean}")

    # Step 4: Load summaries (text and audio)
    text_summaries = load_summaries(experiment_folder, "summary")
    audio_summaries = load_audio_summaries(experiment_folder)

    # Step 5: Concatenate significant summaries (text and audio)
    concatenated_text_summary, num_text_snippets_used = concatenate_significant_summaries(text_significance, text_summaries, text_mean)
    concatenated_audio_summary, num_audio_snippets_used = concatenate_significant_audio(audio_significance, audio_summaries, audio_mean)

    # Step 6: Save concatenated summaries (text and audio)
    save_concatenated_summary(concatenated_text_summary, experiment_folder, "text")
    save_concatenated_audio(concatenated_audio_summary, experiment_folder)

    # Step 7: Apply METEOR metric for text summaries
    all_concatenated_text_summary = "\n".join(text_summaries)
    meteor_text_score = apply_meteor_metric(all_concatenated_text_summary, concatenated_text_summary)

    print("METEOR Score for Text Summaries:")
    print(meteor_text_score)

    # Save METEOR score for text to JSON file
    with open(os.path.join(experiment_folder, "meteor_text_score.json"), "w") as f:
        json.dump({"meteor_score": meteor_text_score}, f, indent=2)

    # Note: METEOR score is not applicable to audio. For audio, you can implement other metrics.

# Run the main process
if __name__ == "__main__":
    process_significance_and_summaries(experiment_folder)
