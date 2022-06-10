import pandas as pd
from my_datasets.my_coherence_dataset import MyCoherenceDataset

def calculate_coherence():
    meansum_path = './my_datasets/meansum_summaries_trimmed.json'
    dataset = MyCoherenceDataset(path_to_dataset=meansum_path)