from bertopic import BERTopic
from typing import Tuple
import pandas as pd
import pickle


def make_predictions(reviews, topic_model_path: str, nr_topics) -> Tuple[BERTopic, tuple]:
    try:
        topic_model = BERTopic.load(topic_model_path)
        with open(topic_model_path + '_predictions', 'rb') as predictions_file:
            predictions = pickle.load(predictions_file)
    except FileNotFoundError:
        topic_model = BERTopic(nr_topics=nr_topics, calculate_probabilities=True)
        topic_model.save(topic_model_path)
        predictions = topic_model.fit_transform(reviews)
        with open(topic_model_path + '_predictions', 'wb') as predictions_file:
            pickle.dump(predictions, predictions_file)
    return (topic_model, predictions)