import numpy as np
import gensim.downloader


def word_to_vec(word):    
    glove_vectors = gensim.downloader.load('glove-twitter-25')
    ##TODO Remove stop tokens from word
    if word.lower() in glove_vectors:
        return glove_vectors[word.lower()]
    else:
        return np.array([0]*25)
