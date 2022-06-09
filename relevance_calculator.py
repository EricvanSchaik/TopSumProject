from helpers.serialization import df_read_json
from my_datasets.my_vectorized_dataset import MyVectorizedDataset
import numpy as np
from helpers.word_to_vec import word_to_vec

def calculate_relevance():
    meansum_dataset = MyVectorizedDataset('./my_datasets/meansum_summaries_trimmed.json')
    pegasus_dataset = MyVectorizedDataset('./my_datasets/pegasus_on_yelp_summaries.json')
    wiki_dataset = MyVectorizedDataset('./my_datasets/wikitext.json')

    distances = []
    subject = word_to_vec('restaurant')

    for vector in meansum_dataset:
        distance = np.linalg.norm(subject - vector.numpy())
        distances.append(distance)
    avg = (sum(distances) / len(distances))
    file = open('./relevance_results.txt', 'a')
    file.write('\nmeansum relevance: ' + str(avg))
    file.close()

    for vector in pegasus_dataset:
        distance = np.linalg.norm(subject - vector.numpy())
        distances.append(distance)
    avg = (sum(distances) / len(distances))
    file = open('./relevance_results.txt', 'a')
    file.write('\npegasus relevance: ' + str(avg))
    file.close()

    for vector in wiki_dataset:
        distance = np.linalg.norm(subject - vector.numpy())
        distances.append(distance)
    avg = (sum(distances) / len(distances))
    file = open('./relevance_results.txt', 'a')
    file.write('\nwiki relevance: ' + str(avg))
    file.close()
