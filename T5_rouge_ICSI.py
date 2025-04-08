import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
import evaluate
import numpy as np

# Updated paths for your text files and summaries
text_chunks_path = '/home/anderson/audio_embedding_research/T5/ICSI_DATA/text_corpus/'
llama_summaries_path = '/home/anderson/audio_embedding_research/T5/ICSI_DATA/summary_llama/'

# Load the ROUGE metric
rouge = evaluate.load("rouge")

# Initialize the T5 model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Summarize text using T5
def summarize_text_t5(text):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Load the chunks and llama summaries
chunk_files = sorted([f for f in os.listdir(text_chunks_path) if f.endswith('.txt')])
llama_summary_files = sorted([f for f in os.listdir(llama_summaries_path) if f.endswith('.txt')])

# Ensure equal number of text chunks and summaries
assert len(chunk_files) == len(llama_summary_files), "Mismatch between text chunks and LLaMA summaries."

# Initialize lists to store ROUGE scores and summaries
rouge1_scores, rouge2_scores, rougeL_scores, rougeLsum_scores = [], [], [], []
t5_summaries = []  # To store the generated T5 summaries

for chunk_file, llama_file in zip(chunk_files, llama_summary_files):
    # Read the text chunk and llama summary
    text_chunk = read_file(os.path.join(text_chunks_path, chunk_file))
    llama_summary = read_file(os.path.join(llama_summaries_path, llama_file))
    
    # Generate T5 summary
    t5_summary = summarize_text_t5(text_chunk)
    t5_summaries.append(t5_summary)
    
    # Compute ROUGE comparing the T5 summary and the LLaMA summary
    rouge_scores = rouge.compute(predictions=[t5_summary], references=[llama_summary])
    
    print(f"ROUGE scores for {chunk_file} (T5 vs LLaMA):")
    print(rouge_scores)
    
    # Accumulate ROUGE scores
    rouge1_scores.append(rouge_scores['rouge1'])
    rouge2_scores.append(rouge_scores['rouge2'])
    rougeL_scores.append(rouge_scores['rougeL'])
    rougeLsum_scores.append(rouge_scores['rougeLsum'])

# Calculate the mean ROUGE scores
mean_rouge1 = np.mean(rouge1_scores)
mean_rouge2 = np.mean(rouge2_scores)
mean_rougeL = np.mean(rougeL_scores)
mean_rougeLsum = np.mean(rougeLsum_scores)

print("\nMean ROUGE scores across all chunks:")
print(f"ROUGE-1: {mean_rouge1}")
print(f"ROUGE-2: {mean_rouge2}")
print(f"ROUGE-L: {mean_rougeL}")
print(f"ROUGE-Lsum: {mean_rougeLsum}")

# If you want to store the T5 summaries in files
output_path = '/home/anderson/audio_embedding_research/T5/ICSI_DATA/summary_t5/'
os.makedirs(output_path, exist_ok=True)
for i, summary in enumerate(t5_summaries):
    with open(os.path.join(output_path, f"summary_t5_{i+1}.txt"), 'w', encoding='utf-8') as file:
        file.write(summary)
