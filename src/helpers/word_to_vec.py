import numpy as np
import gensim.downloader
import string

class WordToVec():

    def __init__(self) -> None:
        self.glove_vectors = gensim.downloader.load('glove-twitter-25')

    def clean_text(self, text: str) -> str:
        return text.translate(str.maketrans('', '', string.punctuation)).lower()

    def word_to_vec(self, word):
        word = self.clean_text(word)
        if word in self.glove_vectors:
            return self.glove_vectors[word]
        else:
            return np.array([0]*25)
