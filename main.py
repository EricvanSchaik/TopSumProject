from multiprocessing.spawn import prepare
from preprocessing import twittertext, wikitext, pegasus
import information_estimator3, information_estimator2
import relevance_calculator

if __name__ == '__main__':
    information_estimator3.estimate_information()