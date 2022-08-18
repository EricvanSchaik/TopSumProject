import pandas as pd
from src.helpers.serialization import df_to_json


def import_summaries():
    dataset = pd.read_json('~/Qsync/Werk/Capgemini/Thesis/Results/meansum_summaries.json')
    dataset.rename(columns={'summary': 'text'}, inplace=True)
    dataset = pd.concat([dataset['categories'], dataset['text']], axis=1)
    dataset['categories'] = dataset['categories'].apply(lambda x: x.split()[0].replace('---', '').replace(',', ''))
    dataset.rename(columns={'categories': 'product_category'}, inplace=True)
    df_to_json(dataset, './data/meansum_summaries.json')
    df_to_json(dataset.head(100), './data/meansum_summaries_trimmed.json')
