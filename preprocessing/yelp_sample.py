import pandas as pd
from helpers.serialization import df_read_json, df_to_json

dataset = df_read_json('C:/Users/ERSCHAIK/Downloads/yelp_sample.json')
dataset = pd.DataFrame(dataset['text'])
df_to_json(dataset, './my_datasets/yelp_text.json')