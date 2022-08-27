import numpy as np
import pandas as pd
import gensim.downloader
from bertopic import BERTopic
from ranking.helpers import get_avg_sents


def rank_reviews(df: pd.DataFrame, nr_topics: int, topic_model: BERTopic, all_products_predictions) -> list:
    product_ids = df['product_id'].unique()
    rankings_per_product = list()
    for id in product_ids:
        reviews = df[df['product_id'] == id]
        review_texts = reviews['review_body']
        
        predictions = all_products_predictions[reviews.index[0]:reviews.index[-1]+1]
        average_sents, compounds = get_avg_sents(review_texts, topic_model.get_topic_info().drop([0]).reset_index(), predictions, nr_topics)

        ## average_sents is now a value between 0 and 2 for each topic

        deviations = list()
        for index, review in enumerate(review_texts):
            deviation_per_topic = list()
            for topic in range(nr_topics):
                deviation_per_topic.append(np.abs(compounds[index] - average_sents[topic]))
            deviations.append(deviation_per_topic)
        
        w2v = gensim.downloader.load('word2vec-google-news-300')

        avg_norms = list()
        for review in review_texts:
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

        ranking_per_topic = list()
        alpha = 0.1
        beta = 0.05
        topic_relevance = predictions
        for topic in range(nr_topics):
            ranking = pd.DataFrame(data={'review': review_texts, 'relevance': np.transpose(topic_relevance)[
                                topic], 'sentiment_deviation': np.transpose(deviations)[topic], 'information': avg_norms})
            ranking['score'] = ranking['relevance'] + alpha * \
                ranking['sentiment_deviation'] + beta*ranking['information']
            ranking = ranking.sort_values(by=['score'], ascending=False)
            ranking.reset_index()
            ranking_per_topic.append(ranking)
        rankings_per_product.append(ranking_per_topic)
    return rankings_per_product
