import pandas as pd
from torch.utils.data import Dataset
import gensim.downloader
import numpy as np
import torch

class AmazonSampleDataset(Dataset):
    
    def __init__(self, labels=False) -> None:
        super().__init__()
        self.PATH_TO_DATASET = 'D:/Users/Eric_/Downloads/sample.json'
        self.dataset = pd.read_json(self.PATH_TO_DATASET, lines=True)
        print(self.dataset)
        
        self.glove_vectors = gensim.downloader.load('glove-twitter-25')

        self.data = ['this text is used to train the recurrent neural network', 'this text is also used to train the recurrent neural network']

        # Create a numpy array of word vectors
        training_vectors = np.concatenate([np.array([self.word_to_vec(word) for word in training_text.split()]) for training_text in self.data])

        # Create a numpy array with all zeros with the same dimension as the training vectors (makes concatenation easier)
        zero_vector = np.array([[0]*25])

        if labels:
            self.data = torch.Tensor(np.concatenate((zero_vector, training_vectors)))
        else:
            self.data = torch.Tensor(np.concatenate((training_vectors, zero_vector)))
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def word_to_vec(self, word):
        ##TODO check for missing words not in glove_vectors
        return self.glove_vectors[word]