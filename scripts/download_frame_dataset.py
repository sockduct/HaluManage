from datasets import load_dataset
import pandas as pd

# 1. Load the dataset from the Hugging Face Hub
# The official name is 'google/frames-benchmark'
# FIX: The ValueError indicated that the configuration name must be 'default'.
# The correct parameters are name="default" and split="test".
dataset = load_dataset("google/frames-benchmark", name="default", split="test")

# 2. Since we specified the split="test" in load_dataset, 'dataset' is now a Dataset object,
# not a DatasetDict, so we no longer need the frames_data = dataset['key'] line.
frames_data = dataset

# 3. Convert to a Pandas DataFrame for easy viewing/manipulation (Optional)
# This will display the first few rows of the data
df = frames_data.to_pandas()
print("Successfully loaded and converted to DataFrame.")
print(df.head())

# 4. Save the data locally as a CSV or JSON file (Optional, but useful for local persistence)
# Save as CSV:
df.to_csv("frames_benchmark_data.csv", index=False)
print("\nData saved locally to frames_benchmark_data.csv")

# Save as JSON (useful since some fields like 'wiki_links' are lists/arrays)
# If you need JSON, you can use:
# df.to_json("frames_benchmark_data.json", orient='records', lines=True)
# print("Data saved locally to frames_benchmark_data.json")

print("Data available in the 'frames_data' variable and saved to frames_benchmark_data.csv.")
