from torch.utils.data import Dataset
import pandas as pd
from os.path import exists
import gensim
import torch

class MyCoherenceDataset(Dataset):

    def __init__(self, path_to_dataset, labels=False) -> None:
        super().__init__()
        print('dataset initialization')
        path_to_vectorized = ''
        if labels:
            path_to_vectorized = path_to_dataset[:-5] + '_labels_vectorized.csv'
        else:
            path_to_vectorized = path_to_dataset[:-5] + '_vectorized.csv'
        if (exists(path_to_vectorized)):
            np_array = pd.read_csv(path_to_vectorized).to_numpy()
        else:
            self.dataset = pd.read_json(path_to_dataset, lines=True)
            self.glove_vectors = gensim.downloader.load('glove-twitter-25')

            
        self.data = torch.Tensor(np_array)
