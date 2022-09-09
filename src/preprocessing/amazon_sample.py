from operator import concat
from re import sub
from datasets import load_dataset
import json
import pandas as pd
from src.helpers.serialization import df_to_json

sample_size = 10000

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
    review_texts = amazon_dataset.remove_columns([c for c in all_column_names if c not in ['product_id', 'review_body', 'product_category']])
    review_texts = review_texts.sort('product_id')
    result = pd.DataFrame()
    freq_counts = pd.DataFrame()
    for i in range(len(review_texts) // sample_size):
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
    result.to_csv('./data/amazon_sorted/most_populair_products.csv', index=False)


def get_n_reviews(n: int):
    amazon_dataset = load_dataset('amazon_us_reviews', 'Electronics_v1_00', split='train')
    all_column_names = list(amazon_dataset.features.keys())
    useful_column_names = ['product_id', 'product_category', 'review_body']
    amazon_dataset = amazon_dataset.remove_columns([c for c in all_column_names if c not in useful_column_names])
    amazon_dataset = amazon_dataset.sort('product_id')
    result = pd.DataFrame(columns=useful_column_names)
    for index in range(len(amazon_dataset)):
        product_ids = amazon_dataset[index:index+n]['product_id']
        if len([x for x in product_ids if x == product_ids[0]]) == n and product_ids[0] not in result['product_id'].tolist():
            result = pd.concat([result, pd.DataFrame.from_dict(amazon_dataset[index:index+n])])
        if len(result) > sample_size:
            break
    df_to_json(result, './data/sorted_amazon/products_' + str(n) + '_reviews.json')
