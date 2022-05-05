import pandas as pd

dataset = pd.read_json('C:/Users/ERSCHAIK/Downloads/yelp_sample.json', lines=True)
dataset = pd.DataFrame(dataset['text'])
dataset.to_json('./my_datasets/yelp_text.json', orient='records', lines=True)