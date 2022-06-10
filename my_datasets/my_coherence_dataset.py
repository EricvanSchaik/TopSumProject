from torch.utils.data import Dataset
from helpers.word_to_vec import word_to_vec
from os.path import exists
import pandas as pd
import json
import numpy as np

class MyCoherenceDataset(Dataset):

    def __init__(self, path_to_dataset) -> None:
        super().__init__()
        path_to_vectorized = path_to_dataset[:-5] + '_sentences.json'
        self.all_vectors = []
        if (exists(path_to_vectorized)):
            with open(path_to_vectorized, 'r') as infile:
                self.all_vectors = json.load(infile)
        else:
            self.dataset = pd.read_json(path_to_dataset, lines=True)
            for review in self.dataset['text'].tolist():
                review_vectors = []
                for sentence in review.split('. '):
                    sentence_vectors = []
                    for word in sentence.split():
                        sentence_vectors.append(word_to_vec(word).tolist())
                    review_vectors.append(sentence_vectors)
                self.all_vectors.append(review_vectors)
            
            with open(path_to_vectorized, 'w') as outfile:
                json.dump(self.all_vectors, outfile)

    def __len__(self):
        return len(self.all_vectors)
    
    def __getitem__(self, index):
        return self.all_vectors[index]

    def get_continuity(self, index):
        sentences = self.all_vectors[index]
        cont_lst = list()
        for i in range(1, len(sentences) - 1):
            words1 = self.get_most_similar_words(sentences[i-1], sentences[i])
            connection1 = (words1[0] + words1[1]) / 2
            words2 = self.get_most_similar_words(sentences[i], sentences[i+1])
            connection2 = (words2[0], words2[1]) / 2
            continuity = (self.similarity(connection1, connection2)) / (len(sentences[i-1]) + len(sentences[i]) + len(sentences[i+1]))
            cont_lst.append(continuity)
        return cont_lst
    
    def similarity(self, vector1, vector2):
        return np.linalg.norm(vector1 - vector2)

    def get_most_similar_words(self, sentence1, sentence2):
        # for word1 in sentence1:
        #     for word2 in sentence2:
                