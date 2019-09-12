# # Evaluation script, Thomas
import os
os.chdir(os.path.dirname(r'C:\Source\Research Repositories\TNBC'))
from datagen import get_data_loaders

from torchvision.datasets import MNIST
from tqdm.autonotebook import tqdm
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import inspect
import time
import torch
from torch import nn, optim
import numpy as np
import pandas as pd

class MnistCAE(nn.Module):
    def __init__(self):
        super(MnistCAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, stride = 2),

            nn.Conv2d(8, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, stride = 2),

            
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
        )

        self.z = nn.AdaptiveAvgPool2d(1)
        self.conv_1 = nn.Conv2d(32, 392, 1)

        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(8, 16, kernel_size =2, stride=2), #14x14
                nn.ConvTranspose2d(16, 32, kernel_size =2, stride=2)
        )

        self.logits = nn.Conv2d(32,1,1)

    def forward(self, x):
        z = self.embed(x)
        z_reshape = self.conv_1(z)
        z_reshape = z_reshape.view(-1, 8, 7,7)
        
        x_up = self.decoder(z_reshape)
        x_tilde = self.logits(x_up)
        return x_tilde

    def embed(self, x):
        x = self.encoder(x)
        z = self.z(x)
        return z

model = MnistCAE()
model.load_state_dict(torch.load(r"C:\Source\Research Repositories\TNBC\MNIST\weights\weights_124.pth"))
model.cuda()
epochs = 1 
train_batch_size = val_batch_size = 256
train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
loss_function = nn.MSELoss()

# optimizer, I've used Adadelta, as it wokrs well without any magic numbers
optimizer = optim.Adam(model.parameters(), lr=0.01)
start_ts = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

losses = []
batches = len(train_loader)
val_batches = len(val_loader)

# progress bar (works in Jupyter notebook too!)
progress = tqdm(enumerate(val_loader), desc="Loss: ", total=val_batches)
# ----------------- TRAINING  -------------------- 
# set model to training
model.eval()

z_train = []
labels_train = []
for i, data in progress:
    X, y = data[0].to(device), data[1].to(device)
    
    # training step for single batch
    model.zero_grad()
    emb = model.embed(X)
    # np.concatenate(y.cpu().numpy(), labels_train)
    labels_train.append(y.cpu().numpy())
    z_train.append(emb.detach().cpu().numpy())
    
    # updating progress bar
    #progress.set_description("Loss: {:.4f}".format(total_loss/(i+1)))
# getting labels and latent representation from validation
labels_train = np.array(labels_train)
labels = np.concatenate(labels_train)
z_train = np.array(z_train)
z = np.ndarray.squeeze(np.concatenate(z_train))

# PCA
pca = PCA(n_components=2).fit_transform(z)
targets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
num_targets = 10000

# t-sne
tsne = TSNE(n_components = 2)
t_space = tsne.fit_transform(z)

# kmeans
n_clusters = 10
kmeans = KMeans(init = 'k-means++',n_clusters=n_clusters).fit(z)
km = kmeans.fit(z)
centroids = kmeans.cluster_centers_

# clustering metrics
# Normalized mutual information score (NMI)
nmi = normalized_mutual_info_score(labels, km.labels_)


# Plotting PCA and KMeans
plt.subplot(2,1,1)
for target in targets:
    idx = np.where(labels==target)
    plt.scatter(pca[idx[0], 0], pca[idx[0], 1])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('PCA of MNIST from LatRep')
plt.legend(targets)

plt.subplot(2,1,2)
for target in targets:
    idx2 = np.where(km.labels_==target)
    plt.scatter(pca[idx2[0], 0], pca[idx2[0], 1])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('PCA with kmeans n=10 cluster labels')
plt.legend(targets)
plt.show()

# Plotting t-sne and KMeans
plt.subplot(2,1,1)
for target in targets:
    idx = np.where(labels==target)
    plt.scatter(t_space[idx[0], 0], t_space[idx[0], 1])
plt.xlabel('t-sne 1')
plt.ylabel('t-sne 2')
plt.title('t-sne of MNIST from LatRep')
plt.legend(targets)

plt.subplot(2,1,2)
for target in targets:
    idx2 = np.where(km.labels_ == target)
    plt.scatter(t_space[idx2[0],0], t_space[idx2[0],1])
plt.xlabel('t-sne 1')
plt.ylabel('t-sne 2')
plt.title('t-sne with kmeans labels')
plt.legend(targets)
plt.show()

plt.subplot(2,1,1)
# fig = plt.figure(figsize=(8,8))
ax=plt.scatter(t_space[:,0], t_space[:,1], c=labels, cmap = 'tab10')
# plt.legend(labels)
# plt.show()

# arbejde videre med: gøre eval.py mere lækkert med funktioner, subplots med PCA vs. t-sne, kigge nærmere på objektive clustering scorer 
hej  = 3
