import pandas as pd
from helpers.serialization import df_to_json

dataset = pd.read_json('C:/Users/ERSCHAIK/Downloads/summaries.json')
dataset.rename(columns={'summary': 'text'}, inplace=True)
dataset = pd.DataFrame(dataset['text'])
df_to_json(dataset, '../../data/yelp_summaries.json')