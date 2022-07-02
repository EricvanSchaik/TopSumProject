import numpy as np
import torch
from my_datasets.my_vectorized_dataset import MyVectorizedDataset

class MyInformationDataset(MyVectorizedDataset):
    
    def __init__(self, path_to_dataset, labels=False) -> None:
        super().__init__(path_to_dataset=path_to_dataset)
 
       # Create a numpy array with all zeros with the same dimension as the training vectors (makes concatenation easier)
        zero_vector = np.array([[0]*25])

        # Since the RNN needs to predict the next word, it is easiest to take as input a zero vector followed by all the word vectors, while the labels can be seen as the opposite
        if labels:
            np_array = np.concatenate((zero_vector, self.training_vectors))
        else:
            np_array = np.concatenate((self.training_vectors, zero_vector))

        self.data = torch.Tensor(np_array)