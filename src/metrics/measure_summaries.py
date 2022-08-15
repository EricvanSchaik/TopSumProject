from src.metrics.information_estimator import test_summaries
from src.metrics.relevance_calculator import calculate_relevance
from src.metrics.word_frequency import count_words

def measure_summaries(path: str, product_category: str):
    print(test_summaries(path))
    print(calculate_relevance(path, product_category))
    print(count_words(path))