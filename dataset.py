from datasets import load_dataset

import librosa
import soundfile


# Load the dataset from Hugging Face
ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")

# Check the dataset structure
print(ds)

sample = ds["train"][0]

clean_audio = sample["clean"]
noisy_audio = sample["noisy"]

print(clean_audio)
