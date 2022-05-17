from torch.utils.data import Dataset
from helpers.serialization import df_read_json, df_to_json
from os.path import exists
import pandas as pd
import gensim.downloader
import numpy as np
import torch

class MyVectorizedDataset(Dataset):

    def __init__(self, path_to_dataset) -> None:
        super().__init__()
        path_to_vectorized = path_to_dataset[:-5] + '_vectorized.json'
        self.training_vectors = []
        if (exists(path_to_vectorized)):
            self.training_vectors = df_read_json(path_to_vectorized).to_numpy()
        else:
            self.dataset = pd.read_json(path_to_dataset, lines=True)
            self.glove_vectors = gensim.downloader.load('glove-twitter-25')

            for review in self.dataset['text'].tolist():
                new_vectors = []
                for word in review.split():
                    if len(new_vectors) == 0:
                        new_vectors = np.array([self.word_to_vec(word)])
                    else:
                        new_vectors = np.concatenate((new_vectors, np.array([self.word_to_vec(word)])))
                if len(self.training_vectors) == 0:
                    self.training_vectors = new_vectors
                else:
                    self.training_vectors = np.concatenate((self.training_vectors, new_vectors))

            df_to_json(pd.DataFrame(self.training_vectors), path_to_dataset[:-5] + '_vectorized.json')

        self.data = torch.Tensor(self.training_vectors)


    def word_to_vec(self, word):
        ##TODO Remove stop tokens from word
        if word.lower() in self.glove_vectors:
            return self.glove_vectors[word.lower()]
        else:
            return np.array([0]*25)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
