import time
import pandas as pd
import pickle
import os
from src.topic_modeling.bert_modeling import make_predictions
from src.ranking.ranking import rank_reviews
from src.summarization.summarizer import summarize
from src.metrics.measure_summaries import measure_summaries
from src.helpers.serialization import df_read_json, df_to_json
from src.preprocessing.distilbart_on_amazon import summarize_amazon

## This is only needed once
# import nltk

# nltk.download()

if __name__ == '__main__':
    start = time.time()

    reviews_path = './data/amazon_sorted/most_populair_products.csv'
    amazon_df = pd.read_csv(reviews_path)
    amazon_df = amazon_df.drop(columns='index').dropna().reset_index()
    reviews = amazon_df['review_body']
    
    topic_model_path = './results/topic_model_all_products'
    nr_topics = 10

    topic_model, predictions = make_predictions(reviews, topic_model_path, nr_topics)
    print('topic model and predictions generated')

    rankings_path = './results/rankings'
    try:
        with open(rankings_path, 'rb') as rankings_file:
            rankings_per_product = pickle.load(rankings_file)
    except FileNotFoundError:
        rankings_per_product = rank_reviews(amazon_df, nr_topics=nr_topics, topic_model=topic_model, all_products_predictions=predictions[1])
        with open(rankings_path, 'wb') as rankings_file:
            pickle.dump(rankings_per_product, rankings_file)
    print('reviews_ranked')

    topsum_path = './results/topsum_summaries.json'
    product_category = amazon_df['product_category'][0]
    try:
        with open(topsum_path, 'rb') as topsum_file:
            final_summaries = df_read_json(topsum_path)
    except FileNotFoundError:
        final_summaries = list()
        for ranking_per_topic in rankings_per_product:
            final_summary = summarize(rankings=ranking_per_topic)
            final_summaries.append(final_summary)
        df_to_json(pd.DataFrame(data={'text': final_summaries, 'product_category': product_category}), path=topsum_path)
    
    results = 'topsum measurements:\n'
    results += measure_summaries(topsum_path)

    results += '\n distilbart measurements:\n'
    if not os.path.exists('./data/distilbart_on_amazon_summaries.json'):
        summarize_amazon()
    results += measure_summaries('./data/distilbart_on_amazon_summaries.json')

    results += '\n meansum measurements:\n'
    results += measure_summaries('./data/meansum_summaries_trimmed.json')
    results_file = open('./results/measurements.txt', 'w')
    results_file.write(results)
    results_file.close()

    end = time.time()
    print('this script took ' + str(end-start) + ' seconds')