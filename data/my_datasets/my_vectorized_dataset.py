from torch.utils.data import Dataset
from src.helpers.word_to_vec import WordToVec
from src.helpers.serialization import df_read_json, df_to_json
from os.path import exists
import pandas as pd

class MyVectorizedDataset(Dataset):

    def __init__(self, path_to_dataset: str, w2v: WordToVec) -> None:
        super().__init__()
        path_to_vectorized = path_to_dataset[:-5] + '_vectorized.json'
        self.review_vectors = []
        if (exists(path_to_vectorized)):
            self.review_vectors = df_read_json(path_to_vectorized).to_numpy()
        else:
            self.dataset = df_read_json(path_to_dataset)
            
            for review in self.dataset['text']:
                if isinstance(review, str):
                    word_vectors = []
                    for word in review.split():
                        if len(word_vectors) == 0:
                            word_vectors = [w2v.word_to_vec(word).tolist()]
                        else:
                            word_vectors += [w2v.word_to_vec(word).tolist()]
                    if len(self.review_vectors) == 0:
                        self.review_vectors = [word_vectors]
                    else:
                        self.review_vectors += [word_vectors]
            print(self.review_vectors)
            df_to_json(pd.DataFrame(self.review_vectors), path_to_dataset[:-5] + '_vectorized.json')

    def __len__(self):
        return len(self.review_vectors)

    def __getitem__(self, index):
        return self.review_vectors[index]
