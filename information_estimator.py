import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from gensim.models import Word2Vec
import gensim.downloader

training_text = 'this text is used to train the recurrent neural network'
test_text =  'we need the information amount of this text'

glove_vectors = gensim.downloader.load('glove-twitter-25')

def word_to_vec(word):
    ##TODO check for missing words not in glove_vectors
    return glove_vectors[word]

data = torch.Tensor([word_to_vec(word) for word in training_text.split()])

INPUT_SIZE = 25
SEQ_LENGTH = 5
HIDDEN_SIZE = 2
NUM_LAYERS = 1
BATCH_SIZE = 4

rnn = nn.RNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, batch_first=True)
inputs = data.view(BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
out, h_n = rnn(inputs)