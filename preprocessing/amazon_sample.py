from datasets import load_dataset
import json
import pandas as pd

def sample_amazon():
    amazon_dataset = load_dataset('amazon_us_reviews', 'Tools_v1_00', split='train')
    all_column_names = list(amazon_dataset.features.keys())
    review_texts = amazon_dataset.remove_columns([c for c in all_column_names if c not in ['product_id', 'review_body']])
    review_texts = review_texts.sort('product_id')
    id_freq = dict()
    review_texts.map(lambda review: count_ids(review, id_freq))
    jsonFile = open('amazon_product_freqs.json', 'w')
    jsonFile.write(json.dumps(id_freq))
    jsonFile.close()
    # freq_df = pd.DataFrame.from_dict(id_freq, orient='index')
    # review_texts_sample = amazon_dataset[:10000]
    # print(review_texts_sample)
    # jsonFile = open('amazon_sample.json', 'w')
    # jsonFile.write(json.dumps(review_texts_sample))
    # jsonFile.close()

def count_ids(review, dictionary):
    product_id = review['product_id']
    if product_id in list(dictionary.keys()):
        dictionary[product_id] += 1
    else:
        dictionary[product_id] = 1