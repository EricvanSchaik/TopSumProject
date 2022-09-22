from os import truncate
from transformers import pipeline
from src.helpers.serialization import df_to_json, df_read_json
import pandas as pd
from math import ceil


def summarize(rankings_per_product, results_path: str) -> list:
    try:
        result = df_read_json(results_path)
    except ValueError:
        product_categories = list()
        
        summarizer = pipeline("summarization", model="google/pegasus-cnn_dailymail", truncation=True) 
        retainment_ratio = 0.2
        full_texts = list()
        product_ids = list()
        i = 0
        for rankings_per_topic in rankings_per_product:
            i += 1
            full_text = ''
            for ranking in rankings_per_topic:
                first_reviews = ranking.head(ceil(len(ranking)*retainment_ratio))['text']
                for review in first_reviews:
                    full_text += '\n' + review
            full_texts.append(full_text)
            product_ids.append(rankings_per_topic[0]['product_id'][0])
            product_categories.append(rankings_per_topic[0]['product_category'][0])
        final_summaries = summarizer(full_texts)
        result = list()
        for summary_dict in final_summaries:
            result.append(summary_dict['summary_text'])
        
        df_to_json(pd.DataFrame(data={'product_id': product_ids, 'text': result, 'product_category': product_categories}), path=results_path)
    return result
