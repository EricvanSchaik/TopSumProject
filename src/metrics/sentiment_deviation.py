import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from src.helpers.serialization import df_read_json


def measure_deviation(summaries_path: str, reviews_path: str) -> str:
    summaries = df_read_json(summaries_path)
    reviews = df_read_json(reviews_path)
    sia = SentimentIntensityAnalyzer()
    deviations = list()
    for _, summary in summaries.iterrows():
        summary_sent = sia.polarity_scores(summary['text'])['compound']
        review_texts = reviews[reviews['product_id'] == summary['product_id']]['review_body']
        deviation = list()
        for text in review_texts:
            deviation.append(np.abs(sia.polarity_scores(text)['compound'] - summary_sent))
        deviations.append(sum(deviation)/len(deviation))
    return str(sum(deviations)/len(deviations))