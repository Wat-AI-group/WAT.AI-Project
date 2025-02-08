from datasets import load_dataset

# Load the VoiceBank-DEMAND-16k dataset from Hugging Face
dataset = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")

# Check available splits in the dataset
print(dataset)

# Access the 'train' split (or any other available split, such as 'test')
train_data = dataset["train"]

# Print the first entry to inspect the data
print(train_data[0])
