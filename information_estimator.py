import numpy as np

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader

import gensim.downloader

training_text = 'this text is used to train the recurrent neural network'
test_text =  'we need the information amount of this text'

glove_vectors = gensim.downloader.load('glove-twitter-25')

def word_to_vec(word):
    ##TODO check for missing words not in glove_vectors
    return glove_vectors[word]

# Create a numpy array of word vectors
training_vectors = np.array([word_to_vec(word) for word in training_text.split()])

# Create a numpy array with all zeros with the same dimension as the training vectors (makes concatenation easier)
zero_vector = np.array([[0]*25])

# Since the RNN needs to predict the next word, it is easiest to take as input a zero vector followed by all the word vectors, while the labels can be seen as the opposite
training_data = DataLoader(torch.Tensor(np.concatenate((zero_vector, training_vectors))), batch_size=5, shuffle=True)
training_labels = DataLoader(torch.Tensor(np.concatenate((training_vectors, zero_vector))), batch_size=5, shuffle=True)

rnn = nn.RNN(25, 25, 1)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=0.01)

lss_avg = []

for epic_batch in range(1, 10):
    lss_per_epoch = []
    for epoch in range(1, 100):
        lss_per_batch = []
        for i, (batch_data, batch_labels) in enumerate(zip(training_data, training_labels)):
            optimizer.zero_grad()
            prob = rnn(batch_data)
            loss = criterion(prob[0], batch_labels)
            lss_per_batch += [loss.item()]
            loss.backward()
            optimizer.step()
        lss_per_epoch += [sum(lss_per_batch) / len(lss_per_batch)]
    lss_avg += [sum(lss_per_epoch) / len(lss_per_epoch)]
print(lss_avg)