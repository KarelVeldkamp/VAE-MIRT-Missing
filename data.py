import torch
import torch.nn.functional as F
import torch.nn.utils.prune
from torch.utils.data import DataLoader, Dataset
import pandas as pd

class ResponseDataset(Dataset):
    """
    Torch dataset for item response data in csv format
    """
    def __init__(self, file_name: str):
        """
        initialize
        :param file_name: path to csv that contains NXI matrix of responses from N people to I items
        """
        # Read csv and ignore rownames
        price_df = pd.read_csv(file_name)
        x = price_df.iloc[:, 1:].values

        self.x_train = torch.tensor(x, dtype=torch.float32)
        missing = torch.isnan(self.x_train)
        self.x_train[missing] = 0
        self.mask = (~missing).int()
        self.x_train = self.x_train.to(device)
        self.mask = self.mask.to(device)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.x_mask[idx]


class SimDataset(Dataset):
    """
    Torch dataset for item response data in numpy array
    """
    def __init__(self, X, device='cpu'):
        """
        initialize
        :param file_name: path to csv that contains NXI matrix of responses from N people to I items
        """
        # Read csv and ignore rownames


        self.x_train = torch.tensor(X, dtype=torch.float32)
        missing = torch.isnan(self.x_train)
        self.x_train[missing] = 0
        self.mask = (~missing).int()
        self.x_train = self.x_train.to(device)
        self.mask = self.mask.to(device)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.mask[idx]

class CSVDataset(Dataset):
    """
    Torch dataset for item response data in numpy array
    """
    def __init__(self, path, device='cpu'):
        """
        initialize
        :param file_name: path to csv that contains NXI matrix of responses from N people to I items
        """
        # Read csv and ignore rownames

        X = pd.read_csv(path, index_col=0).values
        self.x_train = torch.tensor(X, dtype=torch.float32)
        missing = torch.isnan(self.x_train)
        self.mask = (~missing).int()
        self.x_train[self.missing] = 0
        self.x_train = self.x_train.to(device)
        self.mask = self.mask.to(device)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.mask[idx]


class PartialDataset():
    def __init__(self, data, device='cpu'):
        self.data = torch.Tensor(data)
        self.n_max = torch.max(torch.sum(~torch.isnan(self.data), 1))
        self.device = device

    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, idx):
        # output to reconstruct
        output = self.data[idx,:]

        # indicating which ratings are not missing
        mask = torch.Tensor((~torch.isnan(output)).int())

        # save list of movie ids and ratings and replace NA's in output with 0s (will be disregarded in loss using mask)
        item_ids = torch.where(~torch.isnan(output))[0]
        ratings = output[item_ids]
        output = torch.nan_to_num(output,0)

        # convert to tensor and pad with constant values
        item_ids = torch.squeeze(F.pad(item_ids, (self.n_max-ratings.shape[0],0), 'constant', value=0))
        ratings = torch.Tensor(F.pad(ratings, (self.n_max - ratings.shape[0], 0), 'constant', value=2))

        # move tensors to GPU
        item_ids = item_ids.to(self.device)
        ratings = ratings.to(self.device)
        output = output.to(self.device)
        mask = mask.to(self.device)
        return item_ids, ratings, output, mask
