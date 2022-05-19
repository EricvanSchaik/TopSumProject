from my_datasets.my_vectorized_dataset import MyVectorizedDataset

class MyCoherenceDataset(MyVectorizedDataset):

    def __init__(self, path_to_dataset) -> None:
        super().__init__(path_to_dataset=path_to_dataset)
