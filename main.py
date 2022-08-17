import pandas as pd
from src.topic_modeling.bert_modeling import make_predictions
from src.ranking.ranking import rank_reviews
from src.summarization.summarizer import summarize
# from src.metrics.measure_summaries import measure_summaries
import time
from src.helpers.serialization import df_to_json
import pickle

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

    rankings_per_product = rank_reviews(amazon_df, nr_topics=nr_topics, topic_model=topic_model, all_products_predictions=predictions[1])
    with open('./results/rankings', 'wb') as rankings_file:
        pickle.dump(predictions, rankings_file)
    print('reviews_ranked')

    final_summaries = list()
    for ranking_per_topic in rankings_per_product:
        final_summary = summarize(rankings=ranking_per_topic)
        final_summaries.append(final_summary)
    topsum_path = './results/topsum_summaries.json'
    df_to_json(pd.DataFrame(data={'text': final_summaries}), path=topsum_path)
    # measure_summaries(topsum_path, product_df['product_category'][0])
    end = time.time()
    print('this script took ' + str(end-start) + ' seconds')