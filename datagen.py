import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader, Sampler, Subset

def get_data_loaders(train_batch_size, val_batch_size):

    mnist = MNIST(download=True, train=True, root=".").data.float()
    data_transform = Compose([ ToTensor(), Normalize((mnist.mean()/255,), (mnist.std()/255,))])
    train_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=True),
                              batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(MNIST(download=False, root=".", transform=data_transform, train=False),
                            batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader


# class DataGen(Dataset):
#     def __init__(self, file_path):
#         self.path = file_path
#         self.labels = labels
#         self.list_IDs = list_IDs
    
#     def __len__(self):
#         'Denote the total # of samples'
#         len_IDs = len(self.list_IDs)
#         return self.len_IDs
    
#     def __getitem__(item, index):
#         ID = self.list_IDs[index]
#         X = torch.load('data/' + ID + '.tif')
#         y = self.labels[ID]
#         return X, y

if __name__ == '__main__':
    # hej = DataGen('C:\Users\dumle\Desktop\Test\D E19_Thomas GDC images TCGA-C8-A3M7-01Z-00-DX1 846C75F1-2E7E-44F7-B21F-C246141558FA svs')
    hej = 32