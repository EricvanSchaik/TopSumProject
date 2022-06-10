from multiprocessing.spawn import prepare
import information_estimator2, information_estimator
import relevance_calculator
import coherence_calculator

if __name__ == '__main__':
    coherence_calculator.calculate_coherence()