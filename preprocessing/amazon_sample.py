from re import sub
from datasets import load_dataset
import json
import pandas as pd

sample_size = 100000

def sample_amazon():
    amazon_dataset = load_dataset('amazon_us_reviews', 'Tools_v1_00', split='train')
    all_column_names = list(amazon_dataset.features.keys())
    review_texts = amazon_dataset.remove_columns([c for c in all_column_names if c not in ['product_id', 'review_body']])
    review_texts = review_texts.sort('product_id')
    review_texts_sample = review_texts[:10000]
    jsonFile = open('amazon_part.json', 'w')
    jsonFile.write(json.dumps(review_texts_sample))
    jsonFile.close()

def sort_amazon():
    amazon_dataset = load_dataset('amazon_us_reviews', 'Electronics_v1_00', split='train')
    all_column_names = list(amazon_dataset.features.keys())
    review_texts = amazon_dataset.remove_columns([c for c in all_column_names if c not in ['product_id', 'review_body']])
    review_texts = review_texts.sort('product_id')
    result = pd.DataFrame()
    freq_counts = pd.DataFrame()
    for i in range((len(review_texts) // sample_size)):
        part_df = pd.DataFrame(review_texts[i*sample_size:(i+1)*sample_size])
        freq_count = part_df.value_counts(subset=['product_id']).to_frame().reset_index()
        freq_counts = pd.concat([freq_counts, freq_count]).sort_values(by=0, ascending=False)
        freq_counts = freq_counts.reset_index(drop=True)
        relevant_products = list()
        i = 0
        for _, row in freq_counts.iterrows():
            relevant_products.append(row['product_id'])
            i += row[0]
            if i > sample_size:
                break
        result = pd.concat([result, part_df])
        result = result[result['product_id'].isin(relevant_products)]
    result = result.reset_index()
    result.to_csv('./amazon_sorted/most_populair_products.csv', index=False)