from src.helpers.serialization import df_read_json
from data.my_datasets.my_vectorized_dataset import MyVectorizedDataset
import numpy as np
from src.helpers.word_to_vec import WordToVec

def calculate_relevance(path: str) -> np.float64:
    print('start calculating relevance')
    w2v = WordToVec()
    dataset = MyVectorizedDataset(path, w2v)
    print('vectorized dataset generated')
    df = df_read_json(path)
    subjects = df['product_category']
    distances = []
    
    for review_index, review_vectors in enumerate(dataset):
        subject = w2v.word_to_vec(subjects[review_index])
        for word_index, word_vector in enumerate(review_vectors):
            if word_vector is not None:
                distance = np.linalg.norm(subject - np.array(word_vector))
                distances.append(distance)
    avg = (sum(distances) / len(distances))
    print(avg)
    return avg
