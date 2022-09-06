from typing import Tuple
from nltk.sentiment import SentimentIntensityAnalyzer


def get_avg_sents(reviews, topic_model_info, predictions, nr_topics) -> Tuple[list, list]:
    sia = SentimentIntensityAnalyzer()
    compounds = list()
    for review in reviews:
        compounds.append(sia.polarity_scores(review)['compound'] + 1)
    
    sents = [0]*nr_topics
    cum_weights = [0]*nr_topics
    for index, compound in enumerate(compounds):
        for topic, _ in topic_model_info.iterrows():
            weight = predictions[index][topic]
            sents[topic] += weight*compound
            cum_weights[topic] += weight
    average_sents = [0]*nr_topics
    for i in range(nr_topics):
        if cum_weights[i] != 0:
            average_sents[i] = sents[i] / cum_weights[i]
        else:
            average_sents[i] = 0

    return (average_sents, compounds)