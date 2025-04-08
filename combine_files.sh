#!/bin/bash

# Navigate to the corpus root directory
cd ~/audio_embedding_research/amicorpus

# Find all .wav files in subdirectories and save to a text file
find $(pwd) -type f -name "*.wav" > all_wav_files.txt
