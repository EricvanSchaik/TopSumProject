from src.helpers.serialization import df_read_json
from data.my_datasets.my_vectorized_dataset import MyVectorizedDataset
import numpy as np
from src.helpers.word_to_vec import word_to_vec

def calculate_relevance(path: str) -> np.float64:
    dataset = MyVectorizedDataset(path)
    df = df_read_json(path)
    subjects = df['product_category']
    distances = []

    for index, review_vectors in enumerate(dataset):
        subject = word_to_vec(subjects[index])
        for word_vector in review_vectors:
            distance = np.linalg.norm(subject - np.array(word_vector))
            distances.append(distance)
    avg = (sum(distances) / len(distances))
    return avg