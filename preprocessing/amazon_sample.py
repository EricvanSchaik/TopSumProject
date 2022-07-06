from datasets import load_dataset
import json

def sample_amazon():
    amazon_dataset = load_dataset('amazon_us_reviews', 'Wireless_v1_00', split='train')
    all_column_names = list(amazon_dataset.features.keys())
    review_texts = amazon_dataset.remove_columns([c for c in all_column_names if c not in ['product_id', 'review_body']])
    review_texts = review_texts.sort('product_id')
    review_texts_sample = amazon_dataset[:10000]
    jsonFile = open('amazon_sample.json', 'w')
    jsonFile.write(json.dumps(review_texts_sample))
    jsonFile.close()