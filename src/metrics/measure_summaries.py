from src.metrics.information_estimator import test_summaries
from src.metrics.relevance_calculator import calculate_relevance
from src.metrics.word_frequency import count_words

def measure_summaries(path: str) -> str:
    results = ''
    results += ('\t next word prediction: ' + str(test_summaries(path)) + '\n')
    results += ('\t average distance of word vector to topic: ' + str(calculate_relevance(path)) + '\n')
    results += ('\t aggregrated frequences of most frequent words: ' + count_words(path) + '\n')
    return results