import pandas as pd
from src.topic_modeling.bert_modeling import make_predictions
from src.ranking.ranking import rank_reviews
from src.summarization.summarizer import summarize
from src.metrics.measure_summaries import measure_summaries

## This is only needed once
# import nltk

# nltk.download()

if __name__ == '__main__':
    amazon_df = pd.read_csv('./data/amazon_sorted/most_populair_products.csv')
    amazon_df = amazon_df.drop(columns='index').dropna().reset_index()
    product_df = amazon_df[amazon_df['product_id'] == amazon_df.iloc[0]['product_id']]

    reviews = product_df['review_body']
    topic_model, predictions = make_predictions(reviews)

    rankings = rank_reviews(reviews=reviews, nr_topics=10, topic_model=topic_model, predictions=predictions)

    topsum_path = './data/topsum_summaries.json'
    summarize(rankings=rankings, path_to_result=topsum_path)
    measure_summaries(topsum_path)