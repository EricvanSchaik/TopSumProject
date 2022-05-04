import pandas as pd

dataset = pd.read_json('D:/Users/EricvanSchaik/Downloads/summaries.json')
dataset.rename(columns={'summary': 'text'}, inplace=True)
dataset = pd.DataFrame(dataset['text'])
dataset.to_json('D:/Users/EricvanSchaik/Downloads/processed_summaries.json', orient='records', lines=True)
print(pd.read_json('D:/Users/EricvanSchaik/Downloads/processed_summaries.json', lines=True))