import os
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio

'''
# Get the paths to the clean and noisy .wav files
clean_audio_path = sample["clean"]["path"]
noisy_audio_path = sample["noisy"]["path"]

# Check the paths (optional)
print("Clean audio path:", clean_audio_path)
print("Noisy audio path:", noisy_audio_path)

# Define the function to process the .wav files
def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Remove trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Resample to 16kHz
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

# Load and process the audio files
clean_audio = load_wav_16k_mono(clean_audio_path)
noisy_audio = load_wav_16k_mono(noisy_audio_path)

# Check the processed tensors
print("Clean audio shape:", clean_audio.shape)
print("Noisy audio shape:", noisy_audio.shape)
'''
# Absolute path to project
project_dir = "/Users/harisherith/Desktop/WAT.AI Audio Denoising Project/WAT.AI-Project"

# Construct clean and noisy file paths
clean_file = os.path.join(project_dir, "Train", "Voice", "DR1", "FCJF0", "SA1.wav")
noisy_file = os.path.join(project_dir, "Train", "Noise", "Household_Appliance", "Household_Appliance_train.wav")

if not os.path.exists(clean_file) or not os.path.exists(noisy_file):
    print("One or more audio files are missing. Please check the paths.")
    print(f"Expected clean file path: {clean_file}")
    print(f"Expected noisy file path: {noisy_file}")
    exit()  # Exit the script if files are missing


if not os.path.exists(clean_file) or not os.path.exists(noisy_file):
    print("One or more audio files are missing. Please check the paths.")
    exit()

# Function to process .wav files
def load_wav_16k_mono(filename):
    """
    Loads a .wav file, decodes it to mono, and resamples it to 16kHz.
    :param filename: Path to the .wav file
    :return: Tensor containing the audio signal
    """
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Remove trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Resample to 16kHz
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

# Process the audio files
clean_audio = load_wav_16k_mono(clean_file)
noisy_audio = load_wav_16k_mono(noisy_file)

# Check the processed tensors
#print("Clean audio shape:", clean_audio.shape)
#print("Noisy audio shape:", noisy_audio.shape)
file_path = "/Users/harisherith/Desktop/Dataset/Source/Train/Voice/DR3"
print("Exists:", os.path.exists(file_path))
print("Readable:", os.access(file_path, os.R_OK))


# Plot the waveforms
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(clean_audio.numpy())
plt.title("Clean Audio")
plt.subplot(2, 1, 2)
plt.plot(noisy_audio.numpy())
plt.title("Noisy Audio")
plt.show()