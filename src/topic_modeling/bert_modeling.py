from bertopic import BERTopic

def make_predictions(reviews) -> tuple:
    nr_topics = 10
    topic_model = BERTopic(nr_topics=nr_topics, calculate_probabilities=True)

    predictions = topic_model.fit_transform(reviews)
    return (topic_model, predictions)