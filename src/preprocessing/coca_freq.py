import pandas as pd

coca_df = pd.read_csv('wordFrequency.csv')
df = pd.concat([coca_df['lemma'], coca_df['freq']], axis=1)
df.to_csv('coca_freq.csv')