import pandas as pd
from torch.utils.data import Dataset

class AmazonSampleDataset(Dataset):
    
    def __init__(self) -> None:
        super().__init__()
        self.PATH_TO_DATASET = 'C:/Users/ERSCHAIK/Qsync/Werk/Capgemini/Thesis/Dataset/1429_1.csv'
        pd.read_csv(self.PATH_TO_DATASET)

    def __len__(self):
        return 0

    def __getitem__(self, index):
        return 0