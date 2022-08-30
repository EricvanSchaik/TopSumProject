import time
import pandas as pd
import pickle
import os
from src.preprocessing.amazon_sample import get_n_reviews
from src.topic_modeling.bert_modeling import make_predictions
from src.ranking.sentence_ranking import rank_reviews
from src.summarization.summarizer import summarize
from src.metrics.measure_summaries import measure_summaries
from src.helpers.serialization import df_read_json, df_to_json
from src.preprocessing.distilbart_on_amazon import summarize_amazon
import gensim.downloader


## This is only needed once
# import nltk

# nltk.download()

if __name__ == '__main__':
    start = time.time()

    reviews_path = './data/amazon_sorted/products_8_reviews.json'
    
    topic_model_path = './results/topic_model_all_products'

    nr_topics = 10

    # topic_model, predictions = make_predictions(reviews_path, topic_model_path, nr_topics=nr_topics)
    # print('topic model and predictions generated')

    rankings_path = './results/rankings'
    # w2v = gensim.downloader.load('word2vec-google-news-300')
    # rankings_per_product = rank_reviews(results_path=rankings_path, reviews_path=reviews_path, nr_topics=nr_topics, topic_model=topic_model, all_reviews_predictions=predictions[1], w2v=w2v)
    # print('sentences ranked')

    topsum_path = './results/topsum_summaries.json'
    # final_summaries = summarize(reviews_path, rankings_per_product=rankings_per_product, results_path=topsum_path)
    # print('summaries generated')

    results = 'topsum measurements:\n'
    print('measuring topsum')
    results += measure_summaries(topsum_path, reviews_path)

    # results += '\n distilbart measurements:\n'
    # if not os.path.exists('./data/distilbart_on_amazon_summaries.json'):
    #     summarize_amazon()
    # print('measuring distilbart')
    # results += measure_summaries('./data/distilbart_on_amazon_summaries.json', reviews_path)

    # print('measuring coop')
    # results += '\n coop measurements:\n'
    # results += measure_summaries('./data/coop_summaries.json', reviews_path)
    # print(results)
    # results_file = open('./results/measurements.txt', 'w')
    # results_file.write(results)
    # results_file.close()

    end = time.time()
    print('this script took ' + str(end-start) + ' seconds')