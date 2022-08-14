from os import truncate
from transformers import pipeline
from src.helpers.serialization import df_to_json
import pandas as pd


def summarize(rankings, path_to_result: str):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", truncation=True)
    topic_summaries = ''
    for ranking in rankings:
        first_reviews = ranking.head(100)['review']
        full_text = ''
        for review in first_reviews:
            full_text += '\n' + review
        topic_summaries += '\n' + summarizer(full_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    final_summary = summarizer(topic_summaries, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    
    topsum_path = '../data/topsum_summaries.json'
    df_to_json(pd.DataFrame(data={'text': [final_summary]}), path=path_to_result)
