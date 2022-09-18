import pandas as pd
from src.helpers.serialization import df_to_json

def sample_yelp():
    review_dataset = pd.read_json('/media/eric/Linux HDD/Qsync/Werk/Capgemini/Thesis/Dataset/yelp/yelp_sample.json', lines=True)
    review_dataset = pd.concat([review_dataset['business_id'], review_dataset['text']], axis=1)
    review_dataset = review_dataset.sort_values(by=['business_id'])
    business_ids = review_dataset['business_id'].unique()

    business_dataset = pd.read_json('/media/eric/Linux HDD/Qsync/Werk/Capgemini/Thesis/Dataset/yelp/yelp_academic_dataset_business.json', lines=True)
    business_dataset = pd.concat([business_dataset['business_id'], business_dataset['categories']], axis=1)
    business_dataset = business_dataset[business_dataset['business_id'].isin(business_ids)]
    business_dataset = business_dataset.sort_values(by=['business_id'])

    review_dataset = review_dataset.merge(business_dataset, how='inner', on='business_id')
    review_dataset = review_dataset.rename(columns={'business_id': 'product_id', 'text': 'review_body', 'categories': 'product_category'})
    relevant_ids = review_dataset['product_id'].value_counts()
    relevant_ids = relevant_ids[relevant_ids == 8].index
    review_dataset = review_dataset[review_dataset['product_id'].isin(relevant_ids)]
    review_dataset['product_category'] = review_dataset['product_category'].apply(lambda x: x.split(',')[0])
    review_dataset['product_category'] = review_dataset['product_category'].apply(lambda x: x.split()[0])

    df_to_json(review_dataset, './data/sorted_yelp/products_8_reviews.json')
