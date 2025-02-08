from datasets import load_dataset
import os
import requests

# Load the VoiceBank-DEMAND-16k dataset from Hugging Face
dataset = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")

# Define a folder where the audio files will be saved
output_folder = "VoiceBank_Audio_Files"
os.makedirs(output_folder, exist_ok=True)

# Access the 'train' split (or any other available split, such as 'test')
train_data = dataset["train"]

# Iterate through the dataset and save the audio files
for i, example in enumerate(train_data):
    print(train_data[0])  # or test_data[0] if you're working with the test split
    audio_url = example['file']  # Assuming 'file' contains the URL of the audio file
    audio_response = requests.get(audio_url)
    
    # Save the audio file locally
    with open(os.path.join(output_folder, f"audio_{i}.wav"), 'wb') as f:
        f.write(audio_response.content)
    
    if i >= 10:  # Just download the first 10 files for testing
        break

print("Download complete!")
