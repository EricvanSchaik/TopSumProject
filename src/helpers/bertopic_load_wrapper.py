from bertopic import BERTopic

def load_model(path: str) -> BERTopic:
    return BERTopic.load(path)