
from deeplearning.datasets import Dataset

class DMSTForgettingDataset(Dataset):
    def __init__(self, 
        word,
        num_per_epoch,
        **kwargs):

        super().__init__(**kwargs)

        self.word = word
        self.num_per_epoch = num_per_epoch

    def __len__(self):

        if self.dataset_type == Dataset.DatasetType.validate:
            return 1

        return self.num_per_epoch

    def __getitem__(self, idx):

        return self.word, 0

    @staticmethod
    def args(parser):

        parser.add_argument('--word', type=str, required=True)
        parser.add_argument('--num_per_epoch', type=int, default=100)
        super(DMSTForgettingDataset, DMSTForgettingDataset).args(parser)