import pandas as pd

dataset = pd.read_json('C:/Users/ERSCHAIK/Downloads/summaries.json')
dataset.rename(columns={'summary': 'text'}, inplace=True)
dataset = pd.DataFrame(dataset['text'])
dataset.to_json('./my_datasets/yelp_summaries.json', orient='records', lines=True)