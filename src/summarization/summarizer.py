from os import truncate
from transformers import pipeline
from src.helpers.serialization import df_to_json, df_read_json
import pandas as pd
from math import ceil
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch


def summarize(rankings_per_product, results_path: str) -> list:
    try:
        result = df_read_json(results_path)
    except ValueError:
        product_categories = list()
        
        retainment_ratio = 0.2
        full_texts = list()
        product_ids = list()
        i = 0
        for rankings_per_topic in rankings_per_product:
            print('filtering ranking' + str(i))
            i += 1
            full_text = ''
            for ranking in rankings_per_topic:
                first_reviews = ranking.head(ceil(len(ranking)*retainment_ratio))['text']
                for review in first_reviews:
                    full_text += '\n' + review
            full_texts.append(full_text)
            product_ids.append(rankings_per_topic[0]['product_id'][0])
            product_categories.append(rankings_per_topic[0]['product_category'][0])

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
        
        df_to_json(pd.DataFrame(data={'product_id': product_ids, 'text': summaries, 'product_category': product_categories}), path=results_path)
    return result
