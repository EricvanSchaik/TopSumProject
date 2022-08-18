from torch.utils.data import Dataset
from src.helpers.preprocess_text import clean_text
from src.helpers.serialization import df_read_json, df_to_json
from os.path import exists
import pandas as pd
import gensim.downloader
import numpy as np
import torch

class MyVectorizedDataset(Dataset):

    def __init__(self, path_to_dataset: str) -> None:
        super().__init__()
        path_to_vectorized = path_to_dataset[:-5] + '_vectorized.json'
        self.review_vectors = []
        if (exists(path_to_vectorized)):
            self.review_vectors = df_read_json(path_to_vectorized).to_numpy()
        else:
            self.dataset = pd.read_json(path_to_dataset, lines=True)
            self.glove_vectors = gensim.downloader.load('glove-twitter-25')

            for review in self.dataset['text'].tolist():
                word_vectors = []
                for word in review.split():
                    if len(word_vectors) == 0:
                        word_vectors = np.array([self.word_to_vec(word)])
                    else:
                        word_vectors = np.concatenate((word_vectors, np.array([self.word_to_vec(word)])))
                if len(self.review_vectors) == 0:
                    self.review_vectors = [word_vectors]
                else:
                    self.review_vectors.append([word_vectors])

            df_to_json(pd.DataFrame(self.review_vectors), path_to_dataset[:-5] + '_vectorized.json')

    def word_to_vec(self, word):
        word = clean_text(word)
        if word in self.glove_vectors:
            return self.glove_vectors[word]
        else:
            return np.array([0]*25)

    def __len__(self):
        return len(self.review_vectors)

    def __getitem__(self, index):
        return self.review_vectors[index]
