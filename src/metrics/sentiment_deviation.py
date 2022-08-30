import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from src.helpers.serialization import df_read_json


def measure_deviation(summaries_path: str, reviews_path: str) -> str:
    summaries = df_read_json(summaries_path)
    reviews = df_read_json(reviews_path)
    sia = SentimentIntensityAnalyzer()
    deviations = list()
    for summary in summaries.iterrows():
        review_texts = reviews[reviews['product_id'] == id]['review_body']
        deviation = list()
        for text in review_texts:
            deviation.append(sia.polarity_scores(text))
    return ''