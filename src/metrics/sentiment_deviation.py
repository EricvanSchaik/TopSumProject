import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from src.helpers.serialization import df_read_json


def measure_deviation(summaries_path: str, reviews_path: str) -> str:
    summaries = df_read_json(summaries_path)
    reviews = df_read_json(reviews_path)
    product_ids = reviews['product_id'].unique()
    sia = SentimentIntensityAnalyzer()
    deviations = list()
    for index, id in enumerate(product_ids):
        summary_text = summaries['text'][index]
        review_texts = reviews[reviews['product_id'] == id]['review_body']
        deviation = list()
        for text in review_texts:
            deviation.append(sia.polarity_scores(text))
    return ''