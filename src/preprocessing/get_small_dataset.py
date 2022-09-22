import pandas as pd


def get_small_dataset(size: int, type: str):
    full_ds = pd.read_json('./data/sorted_' + type + '/products_8_reviews.json')
    small_ds = full_ds[:size]
    small_ds.to_json('./data/sorted_' + type + '/products_8_reviews_small.json', orient='records')