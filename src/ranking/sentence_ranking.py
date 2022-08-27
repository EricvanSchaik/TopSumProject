import pandas as pd
import numpy as np
from bertopic import BERTopic
from src.ranking.helpers import get_avg_sents
from nltk.sentiment import SentimentIntensityAnalyzer
import gensim.downloader


def rank_reviews(df: pd.DataFrame, nr_topics: int, topic_model: BERTopic, all_reviews_predictions) -> list:
    product_ids = df['product_id'].unique()
    ranking_per_product = list()
    for id in product_ids:
        reviews = df[df['product_id'] == id]
        review_texts = reviews['review_body']
        sentences = list()
        for sentence in review_texts:
            sentences.extend(sentence.split('.'))
        
        predictions = all_reviews_predictions[reviews.index[0]:reviews.index[-1]+1]
        average_sents, _ = get_avg_sents(review_texts, topic_model.get_topic_info().drop([0]).reset_index(), predictions, nr_topics)

        deviations = list()
        sia = SentimentIntensityAnalyzer()
        for sentence in sentences:
            deviation_per_topic = list()
            for topic in range(nr_topics):
                compound = sia.polarity_scores(sentence)['compound'] + 1
                deviation_per_topic.append(np.abs(compound - average_sents[topic]))
            deviations.append(deviation_per_topic)

        w2v = gensim.downloader.load('word2vec-google-news-300')
        
        avg_norms = list()
        for sentence in sentences:
            total_norm = 0
            valid_words = 0
            for word in sentence.split():
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
        predictions_per_sentence = topic_model.transform(sentences)[1]
        for topic in range(nr_topics):
            ranking = pd.DataFrame(data={'sentence': sentences, 'relevance': np.transpose(predictions_per_sentence)[topic], 'sentiment_deviation': np.transpose(deviations)[topic], 'information': avg_norms})
            ranking_per_topic.append(ranking)
            ranking['score'] = ranking['relevance'] + alpha * \
                ranking['sentiment_deviation'] + beta*ranking['information']
            ranking = ranking.sort_values(by=['score'], ascending=False)
            ranking.reset_index()
        ranking_per_product.append(ranking_per_topic)
    return ranking_per_product