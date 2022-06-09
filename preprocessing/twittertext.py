from datasets import load_dataset
from helpers.serialization import df_to_json
import pandas as pd

def generate_tweets():
    dataset = load_dataset('tweet_eval', 'sentiment', split='train')
    tweets = dataset.shuffle()[:1000]
    tweets_df = pd.DataFrame(tweets)
    print(tweets_df)
