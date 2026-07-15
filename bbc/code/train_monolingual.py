from datasets import load_dataset

# Data Preparation
raw_dataset = load_dataset("wikimedia/wikipedia", "20231101.bbc", split="train")
raw_dataset