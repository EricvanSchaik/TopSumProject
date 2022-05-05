import pandas as pd
from torch.utils.data import Dataset
import gensim.downloader
import numpy as np
import torch

class MyCustomDataset(Dataset):
    
    def __init__(self, path_to_dataset, labels=False) -> None:
        super().__init__()
        print('dataset initialization')
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