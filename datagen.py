import os
import torch
import torchvision.transforms as transforms
import pandas as pd
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils import data
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision.datasets import MNIST
from PIL import Image
import numpy as np

def get_data_loaders(dataset, train_batch_size, val_batch_size):
    if dataset == 'MNIST-train':
        mnist = MNIST(download=True, train=True, root=".").data.float()
        data_transform = Compose([ ToTensor()])#, Normalize((mnist.mean()/255,), (mnist.std()/255,))])
        train_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=True),
                                  batch_size=train_batch_size, shuffle=False)
        val_loader = DataLoader(MNIST(download=False, root=".", transform=data_transform, train=False),
                                  batch_size=val_batch_size, shuffle=False)
    elif dataset == 'TNBC':
        print('hej')
        train_loader = val_loader = 1
    return train_loader, val_loader



class DataGen_TNBC(data.Dataset):
    def __init__(self, partition, df, transform = None):
        #self.df = pd.read_csv(df)
        #self.df = self.df.loc[self.df.Group==partition]
        # self.eval = self.df.loc[self.df.Group=='eval']
        dat_path = os.path.abspath(r'Z:\Speciale\TNBC\Data')
        #self.FileID = self.df.FileID
        #self.path = self.df.File
        self.transform = transform 
        folders = os.listdir(dat_path)
        IDs = []
        paths = []
        for folder in folders:
            ID = os.listdir(os.path.join(dat_path, folder))
            for idx in enumerate(ID):
                path = os.path.join(dat_path, folder, idx[1])
                paths.append(path)
            IDs.append(ID)
        self.FileID = np.concatenate(IDs)
        self.path = paths
    
    def __len__(self):
        'Denote the total # of samples'
        # len_IDs = len(self.FileID)
        return len(self.FileID)
    
    def __getitem__(self, idx):
        ID = self.path[idx]
        # X = torch.load(self.path[idx])
        X = Image.open(ID)
        if self.transform is not None:
            X = self.transform(X)

        tag = self.FileID[idx][:-4]
        # y = ToTensor()(y)
        return X, tag

if __name__ == '__main__':

    transform = transforms.Compose([transforms.RandomCrop(size ), transforms.ToTensor()])
    trainer = DataGen_TNBC('train', "C:\Source\Research Repositories\TNBC\data\dataframe_512.csv", transform)
    train_loader = data.DataLoader(trainer, batch_size = 5, shuffle = False)

    test = trainer[0]
    val = DataGen_TNBC('eval', "C:\Source\Research Repositories\TNBC\data\dataframe_512.csv")
    val_loader = data.DataLoader(val, batch_size = 5, shuffle = False)
    hej2 = 32