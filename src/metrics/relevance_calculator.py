from src.helpers.serialization import df_read_json
from data.my_datasets.my_vectorized_dataset import MyVectorizedDataset
import numpy as np
from src.helpers.word_to_vec import WordToVec

def calculate_relevance(path: str) -> np.float64:
    w2v = WordToVec()
    dataset = MyVectorizedDataset(path, w2v)
    df = df_read_json(path)
    subjects = df['product_category']
    distances = []
    
    for summary_index, summary_vectors in enumerate(dataset[10]):
        subject = w2v.word_to_vec(subjects[summary_index])
        for word_vector in summary_vectors:
            if word_vector is not None:
                distance = np.linalg.norm(subject - np.array(word_vector))
                distances.append(distance)
    avg = (sum(distances) / len(distances))
    return avg
