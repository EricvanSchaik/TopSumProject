import pandas as pd
from helpers.serialization import df_read_json, df_to_json

dataset = df_read_json('/home/eric/Qsync/Werk/Capgemini/Thesis/Dataset/yelp/yelp_sample.json')
dataset = pd.DataFrame(dataset['text'])
df_to_json(dataset, './datasets/yelp_text.json')