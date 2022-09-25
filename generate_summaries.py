import time
import os
from src.helpers.word_to_vec import WordToVec
from src.preprocessing.get_small_dataset import get_small_dataset
from src.topic_modeling.bert_modeling import make_predictions
from src.ranking.sentence_ranking import rank_reviews
from src.summarization.summarizer import summarize
from src.metrics.measure_summaries import measure_summaries
from src.preprocessing.textsum_on_reviews import summarize_with_pipeline
import gensim.downloader


if __name__ == '__main__':
    start = time.time()

    dataset = 'amazon'

    reviews_path = './data/sorted_' + dataset + '/products_8_reviews.json'
    
    topic_model_path = './results/' + dataset + '/topic_model_all_products'

    nr_topics = 10

    topic_model, predictions = make_predictions(reviews_path, topic_model_path, nr_topics=nr_topics)
    print('topic model and predictions generated')
    print(time.time() - start)

    rankings_path = './results/' + dataset + '/rankings'
    w2v = WordToVec()
    print(time.time() - start)
    rankings_per_product = rank_reviews(results_path=rankings_path, reviews_path=reviews_path, topic_model=topic_model, all_reviews_predictions=predictions[1], w2v=w2v)
    print('sentences ranked')
    print(time.time() - start)

    topsum_path = './results/' + dataset + '/topsum_summaries.json'
    final_summaries = summarize(rankings_per_product=rankings_per_product, results_path=topsum_path)
    print('summaries generated')