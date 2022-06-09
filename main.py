from multiprocessing.spawn import prepare
from preprocessing import twittertext, wikitext, pegasus
import information_estimator2, information_estimator
import relevance_calculator

if __name__ == '__main__':
    information_estimator.estimate_information()