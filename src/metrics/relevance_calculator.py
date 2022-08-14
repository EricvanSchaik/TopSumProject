from src.helpers.serialization import df_read_json
from data.my_datasets.my_vectorized_dataset import MyVectorizedDataset
import numpy as np
from src.helpers.word_to_vec import word_to_vec

def calculate_relevance(path):
    dataset = MyVectorizedDataset(path)
    distances = []
    subject = word_to_vec('restaurant')

    for vector in dataset:
        distance = np.linalg.norm(subject - vector.numpy())
        distances.append(distance)
    avg = (sum(distances) / len(distances))
    return avg