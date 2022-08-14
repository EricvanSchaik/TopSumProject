from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import gensim.downloader


def rank_reviews(reviews: list, nr_topics: int, topic_model, predictions) -> list:
    sia = SentimentIntensityAnalyzer()
    reviews = reviews.to_list()
    compounds = list()
    for review in reviews:
        compounds.append(sia.polarity_scores(review)['compound'] + 1)

    sents = [0]*nr_topics
    cum_weights = [0]*nr_topics
    topic_model_info = topic_model.get_topic_info().drop([0]).reset_index()
    for index, compound in enumerate(compounds):
        for topic, row in topic_model_info.iterrows():
            weight = predictions[1][index][topic]
            sents[topic] += weight*compound
            cum_weights[topic] += weight
    average_sents = [0]*nr_topics
    for i in range(nr_topics):
        average_sents[i] = sents[i] / cum_weights[i]

    ## average_sents is now a value between 0 and 2 for each topic

    deviations = list()
    for index, review in enumerate(reviews):
        deviation_per_topic = list()
        for topic in range(nr_topics):
            deviation_per_topic.append(np.abs(compounds[index] - average_sents[topic]))
        deviations.append(deviation_per_topic)
    
    w2v = gensim.downloader.load('word2vec-google-news-300')

    avg_norms = list()
    for review in reviews:
        total_norm = 0
        valid_words = 0
        for word in review.split():
            try:
                total_norm += np.linalg.norm(w2v[word])
                valid_words += 1
            except KeyError:
                continue
        try:
            avg_norm = total_norm / valid_words
            avg_norms.append(avg_norm)
        except ZeroDivisionError:
            avg_norms.append(1)

    rankings = list()
    alpha = 0.1
    beta = 0.05
    topic_relevance = predictions[1]
    for topic in range(nr_topics):
        ranking = pd.DataFrame(data={'review': reviews, 'relevance': np.transpose(topic_relevance)[
                            topic], 'sentiment_deviation': np.transpose(deviations)[topic], 'information': avg_norms})
        ranking['score'] = ranking['relevance'] + alpha * \
            ranking['sentiment_deviation'] + beta*ranking['information']
        ranking = ranking.sort_values(by=['score'], ascending=False)
        ranking.reset_index()
        rankings.append(ranking)
    return rankings
