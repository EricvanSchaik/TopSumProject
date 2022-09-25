from cgitb import text
from transformers import pipeline
import pandas as pd
from src.helpers.serialization import df_to_json, df_read_json
import os
from typing import List
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

def summarize_with_pipeline(textsum_path: str, reviews_path: str):
    summarizer = pipeline("summarization", model="google/pegasus-cnn_dailymail", truncation=True)
    amazon_df = df_read_json(reviews_path)
    amazon_df = amazon_df.dropna().reset_index()
    product_ids = amazon_df['product_id'].unique()
    review_summaries = list()
    for index, id in enumerate(product_ids):
        product_reviews = amazon_df[amazon_df['product_id'] == id]['review_body']
        product_reviews.reset_index(drop=True, inplace=True)
        full_text = ''
        for review in product_reviews:
            full_text += '\n' + review
        review_summaries.append(summarizer(full_text)[0]['summary_text'])
        print(str(index) + ' summaries generated')
    results_df = pd.DataFrame.from_dict(summarizer(inputs=review_summaries, max_length=60))
    results_df.rename(columns={'summary_text': 'text'}, inplace=True)
    results_df['product_category'] = amazon_df['product_category'][0]
    results_df['product_id'] = product_ids
    df_to_json(results_df, textsum_path)


def summarize_manually(textsum_path: str, reviews_path: str):
    reviews = pd.read_json(reviews_path)
    reviews = reviews.dropna().reset_index()
    product_ids = reviews['product_id'].unique()
    product_categories = list()
    full_texts = list()
    for id in product_ids:
        products = reviews[reviews['product_id'] == id]
        products.reset_index(inplace=True)
        product_categories.append(products['product_category'][0])
        product_reviews = products['review_body']
        product_reviews.reset_index(drop=True, inplace=True)
        full_text = ''
        for review in product_reviews:
            full_text += '\n' + review
        full_texts.append(full_text)
    
    model_name = "google/pegasus-cnn_dailymail"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    summaries = list()
    for index, text in enumerate(full_texts):
        print('summary ' + str(index))
        batch = tokenizer(text, truncation=True, padding="longest", return_tensors="pt").to(device)
        translated = model.generate(**batch)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        summaries.append(tgt_text[0])
    result = pd.DataFrame.from_dict({'text': summaries, 'product_id': product_ids, 'product_category': product_categories})
    result.to_json(textsum_path, orient='records')