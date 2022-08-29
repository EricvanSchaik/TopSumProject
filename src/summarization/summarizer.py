from os import truncate
from transformers import pipeline
from src.helpers.serialization import df_to_json
import pandas as pd
from math import ceil


def summarize(rankings_per_product) -> list:
    summarizer = pipeline("summarization", truncation=True)
    retainment_ratio = 0.2
    full_texts = list()
    for rankings_per_topic in rankings_per_product:
        full_text = ''
        for ranking in rankings_per_topic:
            first_reviews = ranking.head(ceil(len(ranking)*retainment_ratio))['text']
            for review in first_reviews:
                full_text += '\n' + review
        full_texts.append(full_text)
    final_summaries = summarizer(full_texts)
    result = list()
    for summary_dict in final_summaries:
        result.append(summary_dict['summary_text'])
    return result
