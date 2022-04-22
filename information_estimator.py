import torch
from gensim.models import Word2Vec
import gensim.downloader

training_text = 'this text is used to train the RNN'

test_text =  'the text is used to determine the information amount'

glove_vectors = gensim.downloader.load('glove-twitter-25')

# rnn = torch.nn.RNN()
print(torch.cuda.is_available())