from os import truncate
from transformers import pipeline
from src.helpers.serialization import df_to_json
import pandas as pd
from math import ceil


def summarize(rankings) -> str:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", truncation=True)
    topic_summaries = ''
    retainment_ratio = 0.2
    for ranking in rankings:
        first_reviews = ranking.head(ceil(len(ranking)*retainment_ratio))['text']
        full_text = ''
        for review in first_reviews:
            full_text += '\n' + review
        topic_summaries += '\n' + summarizer(full_text)[0]['summary_text']
    final_summary = summarizer(topic_summaries)[0]['summary_text']
    
    return final_summary
