from multiprocessing.sharedctypes import Value
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
        
        self.glove_vectors = gensim.downloader.load('glove-twitter-25')

        self.data = ['this text is used to train the recurrent neural network', 'this text is also used to train the recurrent neural network']

        # Create a numpy array of word vectors
        # training_vectors = np.concatenate([np.array([self.word_to_vec(word) for word in training_text.split()]) for training_text in self.dataset['text'].tolist()])

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
            self.data = torch.Tensor(np.concatenate((zero_vector, training_vectors)))
        else:
            self.data = torch.Tensor(np.concatenate((training_vectors, zero_vector)))
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def word_to_vec(self, word):
        if word.lower() in self.glove_vectors:
            return self.glove_vectors[word.lower()]
        else:
            return np.array([0]*25)