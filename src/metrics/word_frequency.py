from src.helpers.serialization import df_read_json
import pandas as pd


def count_words(path: str):
    coca_df = pd.read_csv('../data/coca_freq.csv', index_col=0)
    coca_dict = coca_df.to_dict()['freq']

    summaries = df_read_json(path)
    total_freq = 0
    total_words = 0
    for review in summaries['text']:
        for word in review.split():
            if word in coca_dict:
                total_freq += coca_dict[word]
            total_words += 1
    return str(total_freq/total_words)