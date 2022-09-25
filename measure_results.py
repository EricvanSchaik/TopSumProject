import time
import os
from src.helpers.word_to_vec import WordToVec
from src.preprocessing.get_small_dataset import get_small_dataset
from src.topic_modeling.bert_modeling import make_predictions
from src.ranking.sentence_ranking import rank_reviews
from src.summarization.summarizer import summarize
from src.metrics.measure_summaries import measure_summaries
from src.preprocessing.textsum_on_reviews import summarize_reviews
import gensim.downloader

if __name__ == '__main__':
    start = time.time()

    dataset = 'yelp'

    reviews_path = './data/sorted_' + dataset + '/products_8_reviews.json'
    topsum_path = './results/' + dataset + '/topsum_summaries.json'

    w2v = WordToVec()

    results = 'topsum measurements:\n'
    print('measuring topsum')
    results += measure_summaries(topsum_path, reviews_path, w2v)
    results_file = open('./results/' + dataset + '/measurements.txt', 'w')
    results_file.write(results)
    results_file.close()
    print(time.time() - start)

    # results = '\n textsum measurements:\n'

    # textsum_path = './data/textsum/textsum_on_' + dataset + '_summaries.json'
    # if not os.path.exists(textsum_path):
    #     summarize_reviews(textsum_path, reviews_path)
    # print('measuring textsum')
    # results += measure_summaries(textsum_path, reviews_path)
    # results_file = open('./results/' + dataset + '/measurements.txt', 'w')
    # results_file.write(results)
    # results_file.close()
    # print(time.time() - start)

    # print('measuring coop')
    # results = '\n coop measurements:\n'
    # results += measure_summaries('./data/coop/coop_on_' + dataset + '_summaries.json', reviews_path)
    # print(time.time() - start)
    # print(results)
    # results_file = open('./results/' + dataset + '/measurements.txt', 'w')
    # results_file.write(results)
    # results_file.close()

    end = time.time()
    print('this script took ' + str(end-start) + ' seconds')