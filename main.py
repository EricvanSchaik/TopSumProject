from multiprocessing.spawn import prepare
from metrics import information_estimator, relevance_calculator

if __name__ == '__main__':
    relevance_calculator.calculate_relevance()