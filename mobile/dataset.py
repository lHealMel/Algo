import torch
from torch.utils.data import Dataset
from random_data import cr_data


class TrainDataset(Dataset):
    def __init__(self):
        self.x_train, self.y_train = cr_data(type="train")

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_train[idx])
        y = torch.FloatTensor(self.y_train[idx])
        return x, y


class TestDataset(Dataset):
    def __init__(self):
        self.x_test, self.y_test =cr_data(type="test")

    def __len__(self):
        return len(self.x_test)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_test[idx])
        y = torch.FloatTensor(self.y_test[idx])
        return x, y
