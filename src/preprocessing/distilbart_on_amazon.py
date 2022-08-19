from transformers import pipeline
import pandas as pd
from src.helpers.serialization import df_to_json
import os

def summarize_amazon():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", truncation=True)
    amazon_df = pd.read_csv('./data/amazon_sorted/most_populair_products.csv')
    amazon_df = amazon_df.drop(columns='index').dropna().reset_index()
    product_ids = amazon_df['product_id'].unique()
    review_summaries = list()
    for id in product_ids:
        product_reviews = amazon_df[amazon_df['product_id'] == id]['review_body']
        product_reviews.reset_index(drop=True, inplace=True)
        subsummaries = ''
        for i in range(10):
            full_text = ''
            for j in range(100):
                full_text += '\n' + product_reviews[i*100+j]
            subsummaries += '\n' + summarizer(full_text)[0]['summary_text']
        review_summaries.append(summarizer(subsummaries)[0]['summary_text'])
    results_df = pd.DataFrame.from_dict(summarizer(inputs=review_summaries))
    results_df.rename(columns={'summary_text': 'text'}, inplace=True)
    results_df['product_category'] = amazon_df['product_category'][0]
    df_to_json(results_df, os.path.join(os.getcwd(), 'data', 'distilbart_on_amazon_summaries.json'))