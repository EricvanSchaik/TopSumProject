from json import load
import pandas as pd
from src.topic_modeling.bert_modeling import make_predictions
from src.ranking.ranking import rank_reviews
from src.summarization.summarizer import summarize
from src.metrics.measure_summaries import measure_summaries
from src.preprocessing.amazon_sample import sort_amazon
from src.preprocessing.distilbart_on_amazon import summarize_amazon
from src.preprocessing.pegasus_on_yelp import summarize_yelp
from bertopic import BERTopic
from src.helpers.bertopic_load_wrapper import load_model

## This is only needed once
# import nltk

# nltk.download()

if __name__ == '__main__':
    # summarize_amazon()
    summarize_yelp()
    # amazon_df = pd.read_csv('./data/amazon_sorted/most_populair_products.csv')
    # amazon_df = amazon_df.drop(columns='index').dropna().reset_index()
    # product_df = amazon_df[amazon_df['product_id'] == amazon_df.iloc[0]['product_id']]

    # reviews = amazon_df['review_body']
    # topic_model_path = './results/topic_model_all_products'

    # topic_model, predictions = make_predictions(reviews)
    # topic_model.save(topic_model_path)

    # topic_model = load_model(topic_model_path)
    # print(topic_model.get_topic_info())

    # rankings = rank_reviews(reviews=reviews, nr_topics=10, topic_model=topic_model, predictions=predictions)

    # topsum_path = './results/topsum_summaries.json'
    # summarize(rankings=rankings, path_to_result=topsum_path)
    # measure_summaries(topsum_path, product_df['product_category'][0])