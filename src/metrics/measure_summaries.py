from src.metrics.sentiment_deviation import measure_deviation
from src.metrics.information_estimator import test_summaries
from src.metrics.relevance_calculator import calculate_relevance
from src.metrics.word_frequency import count_words

def measure_summaries(summaries_path: str, reviews_path: str, w2v, results_path: str) -> str:
    print('next word prediction')
    results = ('\t next word prediction: ' + str(test_summaries(summaries_path)) + '\n')
    results_file = open(results_path, 'a')
    results_file.write(results)
    results_file.close()
    print('relevance calculation')
    results = ('\t average distance of word vector to topic: ' + str(calculate_relevance(summaries_path, w2v)) + '\n')
    results_file = open(results_path, 'a')
    results_file.write(results)
    results_file.close()
    print('count words')
    results = ('\t aggregrated frequences of most frequent words: ' + count_words(summaries_path) + '\n')
    results_file = open(results_path, 'a')
    results_file.write(results)
    results_file.close()
    print('sentiment deviation calculation')
    results = ('\t sentiment deviation: ' + measure_deviation(summaries_path, reviews_path) + '\n')
    results_file = open(results_path, 'a')
    results_file.write(results)
    results_file.close()
    return results