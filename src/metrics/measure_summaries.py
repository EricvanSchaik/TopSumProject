from src.metrics.rouge_calculation import calculate_rouge
from src.metrics.sentiment_deviation import measure_deviation
from src.metrics.information_estimator import test_summaries
from src.metrics.relevance_calculator import calculate_relevance
from src.metrics.word_frequency import count_words

def measure_summaries(summaries_path: str, reviews_path: str) -> str:
    results = ''
    print('rouge score calculation')
    results += ('\t ROUGE score: ' + calculate_rouge(summaries_path, reviews_path) + '\n')
    print('next word prediction')
    results += ('\t next word prediction: ' + str(test_summaries(summaries_path)) + '\n')
    print('relevance calculation')
    results += ('\t average distance of word vector to topic: ' + str(calculate_relevance(summaries_path)) + '\n')
    print('count words')
    results += ('\t aggregrated frequences of most frequent words: ' + count_words(summaries_path) + '\n')
    print('sentiment deviation calculation')
    # results += ('\t sentiment deviation: ' + measure_deviation(summaries_path, reviews_path) + '\n')
    return results