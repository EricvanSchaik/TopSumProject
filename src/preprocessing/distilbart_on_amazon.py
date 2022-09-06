from transformers import pipeline
import pandas as pd
from src.helpers.serialization import df_to_json, df_read_json
import os

def summarize_amazon(distilbart_path: str, reviews_path: str):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", truncation=True)
    amazon_df = df_read_json(reviews_path)
    amazon_df = amazon_df.dropna().reset_index()
    product_ids = amazon_df['product_id'].unique()
    review_summaries = list()
    for id in product_ids:
        product_reviews = amazon_df[amazon_df['product_id'] == id]['review_body']
        product_reviews.reset_index(drop=True, inplace=True)
        full_text = ''
        for review in product_reviews:
            full_text += '\n' + review
        review_summaries.append(summarizer(full_text)[0]['summary_text'])
    results_df = pd.DataFrame.from_dict(summarizer(inputs=review_summaries, max_length=50))
    results_df.rename(columns={'summary_text': 'text'}, inplace=True)
    results_df['product_category'] = amazon_df['product_category'][0]
    results_df['product_id'] = product_ids
    df_to_json(results_df, distilbart_path)