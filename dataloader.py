from torch.utils.data import DataLoader

class EndovisLoader(DataLoader):
    def __init__(self, dataset, batch_size):
        super().__init__(dataset, batch_size=batch_size)


