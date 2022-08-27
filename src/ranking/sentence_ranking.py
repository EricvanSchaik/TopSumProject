import pandas as pd
import numpy as np
from bertopic import BERTopic
from src.ranking.helpers import get_avg_sents
from nltk.sentiment import SentimentIntensityAnalyzer


def rank_reviews(df: pd.DataFrame, nr_topics: int, topic_model: BERTopic, all_products_predictions) -> list:
    product_ids = df['product_id'].unique()
    rankings_per_product = list()
    for id in product_ids:
        reviews = df[df['product_id'] == id]
        review_texts = reviews['review_body']
        sentences = list()
        for review in review_texts:
            sentences.extend(review.split('.'))
        
        predictions = all_products_predictions[reviews.index[0]:reviews.index[-1]+1]
        average_sents, _ = get_avg_sents(review_texts, topic_model.get_topic_info().drop([0]).reset_index(), predictions, nr_topics)

        deviations = list()
        sia = SentimentIntensityAnalyzer()
        for sentence in sentences:
            deviation_per_topic = list()
            for topic in range(nr_topics):
                compound = sia.polarity_scores(sentence)['compound'] + 1
                deviation_per_topic.append(np.abs(compound - average_sents[topic]))
            deviations.append(deviation_per_topic)

        for topic in range(nr_topics):
            ranking = pd.DataFrame(data={'sentence': sentences, 'sentiment_deviation': np.transpose(deviations)[topic]})
        print(ranking)
        break
    return rankings_per_product