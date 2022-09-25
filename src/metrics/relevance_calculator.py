from src.helpers.serialization import df_read_json
import numpy as np
from src.helpers.word_to_vec import WordToVec

def calculate_relevance(path: str, w2v: WordToVec) -> np.float64:
    df = df_read_json(path)
    subjects = df['product_category']
    distances = []
    
    for summary_index, summary in enumerate(df['text']):
        print('calculating relevance of summary ' + str(summary_index))
        subject = w2v.word_to_vec(subjects[summary_index])
        for word in summary.split():
            word_vector = w2v.word_to_vec(word)
            if word_vector is not None:
                distance = np.linalg.norm(subject - np.array(word_vector))
                distances.append(distance)
    avg = (sum(distances) / len(distances))
    return avg
