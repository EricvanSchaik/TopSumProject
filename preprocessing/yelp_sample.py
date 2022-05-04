import pandas as pd

dataset = pd.read_json('D:/Users/EricvanSchaik/Downloads/yelp_sample.json', lines=True)
dataset = pd.DataFrame(dataset['text'])
dataset.to_json('D:/Users/EricvanSchaik/Downloads/processed_yelp_sample.json', orient='records', lines=True)