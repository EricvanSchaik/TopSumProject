import time
import os
from src.helpers.word_to_vec import WordToVec
from src.preprocessing.get_small_dataset import get_small_dataset
from src.topic_modeling.bert_modeling import make_predictions
from src.ranking.sentence_ranking import rank_reviews
from src.summarization.summarizer import summarize
from src.metrics.measure_summaries import measure_summaries
from src.preprocessing.textsum_on_reviews import summarize_manually, summarize_with_pipeline
import gensim.downloader

if __name__ == '__main__':
    start = time.time()

    dataset = 'amazon'

    reviews_path = './data/sorted_' + dataset + '/products_8_reviews.json'
    topsum_path = './results/' + dataset + '/topsum_summaries.json'

    textsum_path = './data/textsum/textsum_on_' + dataset + '_summaries.json'
    if not os.path.exists(textsum_path):
        summarize_manually(textsum_path, reviews_path)

    w2v = WordToVec()

    results_path = './results/' + dataset + '/measurements.txt'

    results = 'topsum measurements:\n'
    results_file = open(results_path, 'a')
    results_file.write(results)
    results_file.close()
    print('measuring topsum')
    measure_summaries(topsum_path, reviews_path, w2v, results_path)
    print(time.time() - start)

    results = '\n textsum measurements:\n'
    results_file = open(results_path, 'a')
    results_file.write(results)
    results_file.close()
    print('measuring textsum')
    measure_summaries(textsum_path, reviews_path, w2v, results_path)
    print(time.time() - start)

    results = '\n coop measurements:\n'
    results_file = open(results_path, 'a')
    results_file.write(results)
    results_file.close()
    print('measuring coop')
    measure_summaries('./data/coop/coop_on_' + dataset + '_summaries.json', reviews_path, w2v, results_path)
    print(time.time() - start)

    end = time.time()
    print('this script took ' + str(end-start) + ' seconds')