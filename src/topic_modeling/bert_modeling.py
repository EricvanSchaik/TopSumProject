from bertopic import BERTopic
from typing import Tuple
import pandas as pd
import pickle

from src.helpers.serialization import df_read_json


def make_predictions(reviews_path, topic_model_path: str, nr_topics) -> Tuple[BERTopic, tuple]:
    df = df_read_json(reviews_path)
    reviews = df['review_body']
    try:
        topic_model = BERTopic.load(topic_model_path)
        with open(topic_model_path + '_predictions', 'rb') as predictions_file:
            predictions = pickle.load(predictions_file)
    except (FileNotFoundError, RuntimeError):
        topic_model = BERTopic(nr_topics=nr_topics, calculate_probabilities=True)
        predictions = topic_model.fit_transform(reviews)
        topic_model.save(topic_model_path)
        with open(topic_model_path + '_predictions', 'wb') as predictions_file:
            pickle.dump(predictions, predictions_file)
    return (topic_model, predictions)