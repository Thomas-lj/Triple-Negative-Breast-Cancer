import os
import torch
import random
import cv2
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, ToPILImage
from torch.utils import data
from torch.utils.data import DataLoader, Sampler, Subset, BatchSampler
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from PIL import Image
import numpy as np

import utils

def get_data_loaders(dataset, train_batch_size, val_batch_size, transform=None):
    if dataset == 'MNIST':
        mnist = MNIST(download=True, train=True, root=".").data.float()
        # data_transform = Compose([ ToTensor()])#, Normalize((mnist.mean()/255,), (mnist.std()/255,))])
        pre_loader = DataLoader(MNIST(download=True, root=".", transform=transform, train=True),
                                  batch_size=train_batch_size, shuffle=True)
        train_loader = DataLoader(MNIST(download=True, root=".", transform=transform, train=True),
                                  batch_size=train_batch_size, shuffle=False)                          
        val_loader = DataLoader(MNIST(download=True, root=".", transform=transform, train=False),
                                  batch_size=val_batch_size, shuffle=False)
        return pre_loader, train_loader, val_loader
    
    if dataset == 'CIFAR':
        data_transform = Compose([ToTensor()])
        train_loader = DataLoader(CIFAR10(download=True, root='.', transform=data_transform, train=True),
                                    batch_size=train_batch_size, shuffle=False)
        val_loader = DataLoader(CIFAR10(download=False, root='.', transform=data_transform, train=False), 
                                    batch_size=val_batch_size, shuffle=False)
        return train_loader, val_loader
        
def save_mnist(data_dir, classes, n_train, n_val, transforms=None):
    train_data = MNIST(root='.', train=True, transform=transforms)
    val_data = MNIST(root='.', train=False, transform=transforms)
    indicies_train = []
    indicies_val = []
    for i in range(len(train_data.targets)):
        for label in classes:
            if train_data.targets[i] == label:
                indicies_train.append(i)
    n_train = n_train*len(classes)
    np.random.seed(seed=71991)
    indicies_train = np.random.choice(indicies_train, n_train)

    for i in range(len(val_data.targets)):
        for label in classes:
            if val_data.targets[i] == label:
                indicies_val.append(i)
    n_val = n_val*len(classes)

    np.random.seed(seed=71991)
    indicies_val = np.random.choice(indicies_val, n_val)
    train_data = data.Subset(train_data, indicies_train)
    val_data = data.Subset(val_data, indicies_val)
    train_loader = data.DataLoader(train_data, batch_size=1, shuffle=False)
    val_loader = data.DataLoader(val_data, batch_size=1)
    # data_dir = os.path.abspath(r"Z:\Speciale\mnist_128x128\train")
    for num in classes:
        os.makedirs(os.path.join(data_dir, str(num)), exist_ok=True)
    

    for idx, (image, label) in enumerate(train_loader):
        img = image.squeeze(0).expand(3,-1,-1)*255
        img = np.uint8(img.numpy())
        img = np.moveaxis(img, 0,2)
        filename = os.path.join(data_dir, str(int(label)), 'im_'+str(idx) + '.tif')
        im = Image.fromarray(img)
        im.save(filename)
    return 

class DataGen_Tissues(object):
    def __init__(self, dat_path, transform = None, tissues=None):
        self.seed = 71991
        self.transform = transform
        if not tissues:
            tissues = os.listdir(dat_path)
        IDs = []
        paths = []
        labels = []

        if "Tissues" in dat_path:
            for label, tissue in enumerate(tissues):
                ID = os.listdir(os.path.join(dat_path, tissue))
                for idx in ID:
                    images = os.listdir(os.path.join(dat_path, tissue, idx))
                    for im in images:
                        image = os.path.join(dat_path, tissue, idx, im)
                        labels.append(label)
                        paths.append(image)
        if "mnist_128x128" in dat_path:
            for number in tissues:
                IDs = os.listdir(os.path.join(dat_path, number))
                for ID in IDs:
                    img = os.path.join(dat_path, number, ID)
                    paths.append(img)
                    labels.append(int(number))
        random.seed(71991)
        dat = list(zip(labels, paths))
        random.shuffle(dat)
        labels, paths = zip(*dat)
        self.labels = labels       
        self.path = paths

    def __len__(self):
        len_IDs = len(self.labels)
        return len_IDs

    def __getitem__(self, idx):
        ID = self.path[idx]
        X = Image.open(ID)
        if self.transform is not None:
            X = self.transform(X)
        y = self.labels[idx]
        return X, y

class ManualRandomSampler(torch.utils.data.RandomSampler):
    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.last_epoch = None
        super().__init__(data_source, replacement=False, num_samples=None)
    
    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        if self.last_epoch is None:
            self.last_epoch = torch.randperm(n).tolist()
  
        return iter(self.last_epoch)

    def shuffle(self):
        random.shuffle(self.last_epoch)

class DataGen_TNBC(data.Dataset):
    def __init__(self, dat_path, transform = None):
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
        IDs = np.concatenate(IDs)
        random.seed(71991)
        dat = list(zip(IDs, paths))
        random.shuffle(dat)
        IDs, paths = zip(*dat)    
        self.FileID = IDs
        self.path = paths
    
    def __len__(self):
        'Denote the total # of samples'
        return len(self.FileID)
    
    def __getitem__(self, idx):
        ID = self.path[idx]
        X = Image.open(ID)
        if self.transform is not None:
            X = self.transform(X)
        tag = self.FileID[idx][:-4]
        return X, tag

class DataGen_Tumor(data.Dataset):
    def __init__(self, dat_path, n, transform=None):
        self.transform = transform
        folders = os.listdir(dat_path)
        IDs = []
        paths = []
        img_len = []
        patient = []
        n_img =[]
        tot_ID = []
        for folder in folders:
            ID = os.listdir(os.path.join(dat_path, folder))
            tot_ID.append(ID)
            if len(ID) > n:
                np.random.seed(71991)
                samples = np.random.choice(len(ID), n, replace=False)
                ID = np.array(ID)[samples]
            for idx in enumerate(ID):
                path = os.path.join(dat_path, folder, idx[1])
                paths.append(path)
            IDs.append(ID)
        IDs = np.concatenate(IDs)
        dat = list(zip(IDs, paths))
        random.seed(71991)
        random.shuffle(dat)
        IDs, paths = zip(*dat)    
        self.FileID = IDs
        self.path = paths

    def __len__(self):
        'Denote the total # of samples'
        return len(self.FileID)
    
    def __getitem__(self, idx):
        ID = self.path[idx]
        X = Image.open(ID)
        if self.transform is not None:
            X = self.transform(X)
        tag = self.path[idx].split('\\')[-2][41:55] + self.path[idx].split('\\')[-1][13:-4]
        # tag = self.FileID[idx][:-4]
        return X, tag

if __name__ == '__main__':
    # TNBC no labels
    # transform = transforms.Compose([transforms.CenterCrop(128), transforms.ToTensor()])
    # train_path = os.path.abspath(r"Z:\Speciale\sampling_train")
    # val_path = os.path.abspath(r"Z:\Speciale\sampling_eval")
    # trainer = DataGen_TNBC(train_path, transform)
    # sampler = ManualRandomSampler(trainer.path)
    # train_loader = data.DataLoader(trainer, sampler=sampler, batch_size=32)
    # Tumor, no labels
    # TO-DO: remove outliers that are very white
    # img_path = os.path.abspath(r"Z:\Speciale\Tumor\D E19_Thomas TNBC - included slides TCGA-B6-A0RE-01Z-00-DX1 8933BDB4-EF8D-41FA-9A50-05BB3AB976E5 svs")
    # ID = os.listdir(img_path)
    # thresholds = []
    # for k in ID:
    #     img = cv2.imread(os.path.join(img_path, k), 0)
    #     thres = utils.otsu(img)
    #     thresholds.append(thres)

    transform = transforms.Compose([ 
                                transforms.CenterCrop(128),
                                # transforms.RandomHorizontalFlip(),
                                ToTensor()])
    tumor_path1 = os.path.abspath(r"Z:\Speciale\Tumor_10x_thumb_vahadane_norm")
    tumor_path2 = os.path.abspath(r"Z:\Speciale\Tumor_10x_thumb_macenko_norm")
    # patients1 = os.listdir(tumor_path1)
    # patients2 = os.listdir(tumor_path2)
    # n_img1 = 0
    # n_img2 = 0
    # for k in patients1:
    #     hej = len(os.listdir(os.path.join(tumor_path1, k)))
    #     hej2 = len(os.listdir(os.path.join(tumor_path2, k)))
    #     if hej!=hej2:
    #         print(k)
    #     n_img1 = n_img1 + hej
    #     n_img2 = n_img2 + hej2

    tumor_trainer1 = DataGen_Tumor(tumor_path1, 400, transform)
    tumor_trainer2 = DataGen_Tumor(tumor_path2, 400, transform)
    tumor_sampler = ManualRandomSampler(tumor_trainer.path)
    train_loader = data.DataLoader(tumor_trainer, sampler=tumor_sampler, batch_size=32)
    # for data in train_loader:
    #     hej, med = data

    # Tissues 4 classes
    tis_train_path = os.path.abspath(r"Z:\Speciale\Tissues\Train")
    tis_val_path = os.path.abspath(r"Z:\Speciale\Tissues\Eval")
    tis_transform = transforms.Compose([transforms.CenterCrop(128), transforms.ToTensor()])
    tis_trainer = DataGen_Tissues(tis_train_path, tis_transform, ['Lymphocytes', 'Tumor'])
    sampler = ManualRandomSampler(tis_trainer)
    tis_train_loader = data.DataLoader(tis_trainer, sampler=sampler, batch_size=32)

    # MNIST
    pre, train, val = get_data_loaders('MNIST_train', 32, 32)
    data_transform = Compose([ ToTensor()])
    test_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=True),
                                  batch_size=32, shuffle=False)
    sampler_mnist = ManualRandomSampler(test_loader.dataset.data)
    # mnist_train_loader = data.DataLoader(train, sampler=sampler_mnist, batch_size=32)
    hej = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=True),
                                batch_size=32, shuffle=False, sampler=sampler_mnist)
    tissues = ['Lymphocytes', 'Tumor']
    
    # Save MNIST to 128x128:
    data_transform = Compose([
                                    Resize((128, 128)),
                                    ToTensor(), 
                                    ])
    data_dir = os.path.abspath(r"Z:\Speciale\mnist_128x128\train")
    save_mnist(data_dir, [4, 5, 8, 9], 1564, 782, data_transform)

    mnist_dir = os.path.abspath(r"Z:\Speciale\mnist_128x128\train")
    mnist_transform = transforms.Compose([transforms.CenterCrop(128), transforms.ToTensor()])
    mnist_trainer = DataGen_Tissues(mnist_dir, mnist_transform)
    sampler = ManualRandomSampler(mnist_trainer)
    mnist_train_loader = data.DataLoader(mnist_trainer, sampler=sampler, batch_size=32)
    hej = 3