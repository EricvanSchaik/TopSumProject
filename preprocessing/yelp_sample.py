import pandas as pd
from helpers.serialization import df_read_json, df_to_json

dataset = df_read_json('/home/eric/Qsync/Werk/Capgemini/Thesis/Dataset/yelp/yelp_sample.json')
dataset = pd.concat([dataset['business_id'], dataset['text']], axis=1)
dataset = dataset.sort_values(by=['business_id'])
df_to_json(dataset, './my_datasets/yelp_text.json')