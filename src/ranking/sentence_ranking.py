import pandas as pd
import numpy as np
from bertopic import BERTopic
from src.ranking.helpers import get_avg_sents
from src.helpers.serialization import df_read_json
from nltk.sentiment import SentimentIntensityAnalyzer
import pickle

def rank_reviews(results_path, reviews_path, topic_model: BERTopic, all_reviews_predictions, w2v) -> list:
    nr_topics = len(topic_model.get_topic_info())-1
    try:
        with open(results_path, 'rb') as rankings_file:
            rankings_per_product = pickle.load(rankings_file)
    except FileNotFoundError:
        amazon_df = df_read_json(reviews_path)
        product_ids = amazon_df['product_id'].unique()
        rankings_per_product = list()
        for id in product_ids:
            reviews = amazon_df[amazon_df['product_id'] == id]
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
            alpha = -0.1
            beta = 0.05
            predictions_per_sentence = topic_model.transform(sentences)[1]
            for topic in range(nr_topics):
                ranking = pd.DataFrame(data={'product_id': id, 'product_category': reviews['product_category'][0], 'text': sentences, 'relevance': np.transpose(predictions_per_sentence)[topic], 'sentiment_deviation': np.transpose(deviations)[topic], 'information': avg_norms})
                ranking['score'] = ranking['relevance'] + alpha * \
                    ranking['sentiment_deviation'] + beta*ranking['information']
                ranking = ranking.sort_values(by=['score'], ascending=False)
                ranking.reset_index()
                ranking_per_topic.append(ranking)
            rankings_per_product.append(ranking_per_topic)
        with open(results_path, 'wb') as rankings_file:
            pickle.dump(rankings_per_product, rankings_file)
    return rankings_per_product