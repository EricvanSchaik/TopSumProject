from datasets import load_dataset
import pandas as pd

def sample_amazon():
    dataset = load_dataset('amazon_us_reviews', 'Wireless_v1_00', split='train')