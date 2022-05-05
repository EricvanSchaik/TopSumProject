import pandas as pd
from torch.utils.data import Dataset
import gensim.downloader
import numpy as np
import torch
from os.path import exists

class MyInformationDataset(Dataset):
    
    def __init__(self, path_to_dataset, labels=False) -> None:
        super().__init__()
        print('dataset initialization')
        path_to_vectorized = ''
        if labels:
            path_to_vectorized = path_to_dataset[:-5] + '_labels_vectorized.csv'
        else:
            path_to_vectorized = path_to_dataset[:-5] + '_vectorized.csv'
        if (exists(path_to_vectorized)):
            np_array = pd.read_csv(path_to_vectorized, index_col=0).to_numpy()
        else:
            self.dataset = pd.read_json(path_to_dataset, lines=True)
            self.glove_vectors = gensim.downloader.load('glove-twitter-25')

            training_vectors = []
            for review in self.dataset['text'].tolist():
                new_vectors = []
                for word in review.split():
                    if len(new_vectors) == 0:
                        new_vectors = np.array([self.word_to_vec(word)])
                    else:
                        new_vectors = np.concatenate((new_vectors, np.array([self.word_to_vec(word)])))
                if len(training_vectors) == 0:
                    training_vectors = new_vectors
                else:
                    training_vectors = np.concatenate((training_vectors, new_vectors))

            # Create a numpy array with all zeros with the same dimension as the training vectors (makes concatenation easier)
            zero_vector = np.array([[0]*25])

            # Since the RNN needs to predict the next word, it is easiest to take as input a zero vector followed by all the word vectors, while the labels can be seen as the opposite
            if labels:
                np_array = np.concatenate((zero_vector, training_vectors))
                pd.DataFrame(np_array).to_csv(path_to_dataset[:-5] + '_labels_vectorized.csv')
            else:
                np_array = np.concatenate((training_vectors, zero_vector))
                pd.DataFrame(np_array).to_csv(path_to_dataset[:-5] + '_vectorized.csv')
        self.data = torch.Tensor(np_array)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def word_to_vec(self, word):
        ##TODO Remove stop tokens from word
        if word.lower() in self.glove_vectors:
            return self.glove_vectors[word.lower()]
        else:
            return np.array([0]*25)