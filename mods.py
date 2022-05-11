# Thomas Leicht Jensen, Sep 2019. Triple negative breast cancer
import torch
import copy
import math
from torch import nn, optim
from sklearn.mixture import GaussianMixture
from torch.autograd import Variable
from sklearn.utils.linear_assignment_ import linear_assignment
from torch.nn import functional as F
import numpy as np
import torch.distributions as dist
import torchvision.models as nmodels

# simple CAE for MNIST dataset
class MnistCAE(nn.Module):
    def __init__(self):
        super(MnistCAE, self).__init__()
        # print(input_shape)
        # encoder part in nn.sequential form:
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

        self.conv_1 = nn.Conv2d(32, 392, 1)
        
        # decoder part in nn.sequential form
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(8, 16, kernel_size =2, stride=2), #14x14
                nn.ConvTranspose2d(16, 32, kernel_size =2, stride=2)
        )
        self.z = nn.AdaptiveAvgPool2d(1)

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
        
# Convolutional autoencoder directly from DCEC article
class CAE_3(nn.Module):
    def __init__(self, input_shape=[28,28,1], num_clusters=10, z_dim=10, filters=[32, 64, 128], leaky=True, neg_slope=0.01, activations=False, bias=True):
        super(CAE_3, self).__init__()
        self.activations = activations
        # bias = True
        self.pretrained = False
        self.num_clusters = num_clusters
        self.input_shape = input_shape
        self.filters = filters
        self.conv1 = nn.Conv2d(input_shape[2], filters[0], 5, stride=2, padding=2, bias=bias)
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
        self.conv3 = nn.Conv2d(filters[1], filters[2], 3, stride=2, padding=0, bias=bias)
        lin_features_len = ((input_shape[0]//2//2-1) // 2) * ((input_shape[0]//2//2-1) // 2) * filters[2]
        self.embedding = nn.Linear(lin_features_len, z_dim, bias=bias)
        self.deembedding = nn.Linear(z_dim, lin_features_len, bias=bias)
        out_pad = 1 if input_shape[0] // 2 // 2 % 2 == 0 else 0
        self.deconv3 = nn.ConvTranspose2d(filters[2], filters[1], 3, stride=2, padding=0, output_padding=out_pad, bias=bias)
        out_pad = 1 if input_shape[0] // 2 % 2 == 0 else 0
        self.deconv2 = nn.ConvTranspose2d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad, bias=bias)
        out_pad = 1 if input_shape[0] % 2 == 0 else 0
        self.deconv1 = nn.ConvTranspose2d(filters[0], input_shape[2], 5, stride=2, padding=2, output_padding=out_pad, bias=bias)
        self.clustering = ClusterlingLayer(z_dim, num_clusters)
        # ReLU copies for graph representation in tensorboard
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.conv3(x)
        if self.activations:
            x = self.sig(x)
        else:
            x = self.relu3_1(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x) # embedding converts from 1152 to 10 dimensions
        extra_out = x
        clustering_out = self.clustering(x)
        x = self.deembedding(x) # deembedding converts from 10 to 1152 again
        x = self.relu1_2(x)
        x = x.view(x.size(0), self.filters[2], ((self.input_shape[0]//2//2-1) // 2), ((self.input_shape[0]//2//2-1) // 2))
        x = self.deconv3(x)
        x = self.relu2_2(x)
        x = self.deconv2(x)
        x = self.relu3_2(x)
        x = self.deconv1(x)
        if self.activations:
            x = self.tanh(x)
        return x, clustering_out, extra_out

    def embed(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.conv3(x)
        if self.activations:
            x = self.sig(x)
        else:
            x = self.relu3_1(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return x

class TNBC_CAE_3(nn.Module):
    def __init__(self, input_shape=[3,512,512], z_dim=512, num_clusters=30, filters=[32, 64, 128], leaky=True, neg_slope=0.01, activations=False, bias=True):
        super(TNBC_CAE_3, self).__init__()
        self.activations = activations
        # bias = True
        self.pretrained = False
        self.num_clusters = num_clusters
        self.input_shape = input_shape
        self.filters = filters
        self.z_dim = z_dim

        self.conv1 = nn.Conv2d(input_shape[0], filters[0], 5, stride=2, padding=2, bias=bias)
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
        self.conv3 = nn.Conv2d(filters[1], filters[2], 3, stride=2, padding=1, bias=bias)
        lin_features_len = ((input_shape[1]//2//2) // 2) * ((input_shape[1]//2//2) // 2) * filters[2]
        self.embedding = nn.Linear(lin_features_len, z_dim, bias=bias)
        self.deembedding = nn.Linear(z_dim, lin_features_len, bias=bias)
        out_pad = 1 if input_shape[1] // 2 // 2 % 2 == 0 else 0
        self.deconv3 = nn.ConvTranspose2d(filters[2], filters[1], 3, stride=2, padding=1, output_padding=out_pad, bias=bias)
        out_pad = 1 if input_shape[1] // 2 % 2 == 0 else 0
        self.deconv2 = nn.ConvTranspose2d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad, bias=bias)
        out_pad = 1 if input_shape[1] % 2 == 0 else 0
        self.deconv1 = nn.ConvTranspose2d(filters[0], input_shape[0], 5, stride=2, padding=2, output_padding=out_pad, bias=bias)
        self.clustering = ClusterlingLayer(z_dim, num_clusters)
        # ReLU copies for graph representation in tensorboard
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.conv3(x)
        if self.activations:
            x = self.sig(x)
        else:
            x = self.relu3_1(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x) # embedding converts from image dependent size to 512 dimensions
        extra_out = x
        clustering_out = self.clustering(extra_out)
        x = self.deembedding(x) # deembedding converts from 512 to image dependent size again
        x = self.relu1_2(x)
        x = x.view(x.size(0), self.filters[2], ((self.input_shape[1]//2//2) // 2), ((self.input_shape[1]//2//2) // 2))
        x = self.deconv3(x)
        x = self.relu2_2(x)
        x = self.deconv2(x)
        x = self.relu3_2(x)
        x = self.deconv1(x)
        # x = self.sig(x) 
        return x, clustering_out, extra_out

class TNBC_CAE_bn3(nn.Module):
    def __init__(self, input_shape=[3,128,128], z_dim=512, num_clusters=30, filters=[32, 64, 128], leaky=True, neg_slope=0.01, activations=False, bias=True):
        super(TNBC_CAE_bn3, self).__init__()
        self.activations=activations
        self.pretrained = False
        self.num_clusters = num_clusters
        self.input_shape = input_shape
        self.filters = filters
        self.z_dim = z_dim

        self.conv1 = nn.Conv2d(input_shape[0], filters[0], 5, stride=2, padding=2, bias=bias)
        self.bn1_1 = nn.BatchNorm2d(filters[0])
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
        self.bn2_1 = nn.BatchNorm2d(filters[1])
        self.conv3 = nn.Conv2d(filters[1], filters[2], 3, stride=2, padding=1, bias=bias)
        lin_features_len = ((input_shape[1]//2//2) // 2) * ((input_shape[1]//2//2) // 2) * filters[2]
        self.embedding = nn.Linear(lin_features_len, z_dim, bias=bias)
        self.deembedding = nn.Linear(z_dim, lin_features_len, bias=bias)
        out_pad = 1 if input_shape[1] // 2 // 2 % 2 == 0 else 0
        self.deconv3 = nn.ConvTranspose2d(filters[2], filters[1], 3, stride=2, padding=1, output_padding=out_pad, bias=bias)
        out_pad = 1 if input_shape[1] // 2 % 2 == 0 else 0
        self.bn3_2 = nn.BatchNorm2d(filters[1])
        self.deconv2 = nn.ConvTranspose2d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad, bias=bias)
        out_pad = 1 if input_shape[1] % 2 == 0 else 0
        self.bn2_2 = nn.BatchNorm2d(filters[0])
        self.deconv1 = nn.ConvTranspose2d(filters[0], input_shape[0], 5, stride=2, padding=2, output_padding=out_pad, bias=bias)
        self.clustering = ClusterlingLayer(z_dim, num_clusters)
        # ReLU copies for graph representation in tensorboard
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.bn1_1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.bn2_1(x)
        x = self.conv3(x)
        if self.activations:
            x = self.sig(x)
        else:
            x = self.relu3_1(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        extra_out = x
        clustering_out = self.clustering(extra_out)
        
        x = self.deembedding(x)
        x = self.relu1_2(x)
        x = x.view(x.size(0), self.filters[2], ((self.input_shape[1]//2//2) // 2), ((self.input_shape[1]//2//2) // 2))
        x = self.deconv3(x)
        x = self.relu2_2(x)
        x = self.bn3_2(x)
        x = self.deconv2(x)
        x = self.relu3_2(x)
        x = self.bn2_2(x)
        x = self.deconv1(x)
        x = self.sig(x)
        return x, clustering_out, extra_out

class TNBC_CAE_4(nn.Module):
    def __init__(self, input_shape=[3,512,512], num_clusters=30, filters=[32, 64, 128, 256], leaky=True, neg_slope=0.01, activations=False, bias=True):
        super(TNBC_CAE_4, self).__init__()
        self.activations = activations
        self.pretrained = False
        self.num_clusters = num_clusters
        self.input_shape = input_shape
        self.filters = filters
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)

        self.conv1 = nn.Conv2d(input_shape[0], filters[0], 5, stride=2, padding=2, bias=bias)
        self.conv2 = nn.Conv2d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
        self.conv3 = nn.Conv2d(filters[1], filters[2], 5, stride=2, padding=2, bias=bias)
        self.conv4 = nn.Conv2d(filters[2], filters[3], 3, stride=2, padding=1, bias=bias)

        lin_features_len = ((input_shape[1] // 2 // 2 // 2) // 2) * ((input_shape[1] // 2 // 2 // 2) // 2) * \
                           filters[3]
        self.embedding = nn.Linear(lin_features_len, 512, bias=bias)
        self.clust = nn.Linear(512, num_clusters, bias=bias)
        self.deembedding = nn.Linear(512, lin_features_len, bias=bias)
        out_pad = 1 if input_shape[1] // 2 // 2 // 2 % 2 == 0 else 0
        self.deconv4 = nn.ConvTranspose2d(filters[3], filters[2], 3, stride=2, padding=1, output_padding=out_pad,
                                          bias=bias)
        out_pad = 1 if input_shape[1] // 2 // 2 % 2 == 0 else 0
        self.deconv3 = nn.ConvTranspose2d(filters[2], filters[1], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        out_pad = 1 if input_shape[1] // 2 % 2 == 0 else 0
        self.deconv2 = nn.ConvTranspose2d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        out_pad = 1 if input_shape[1] % 2 == 0 else 0
        self.deconv1 = nn.ConvTranspose2d(filters[0], input_shape[0], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.clustering = ClusterlingLayer(num_clusters, num_clusters)
        # ReLU copies for graph representation in tensorboard
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu4_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.relu4_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):   # [?,3,512,512]
        x = self.conv1(x)   # [?,32,256,256]
        x = self.relu1_1(x)
        x = self.conv2(x)   # [?,64,128,128]
        x = self.relu2_1(x)
        x = self.conv3(x)   # [?,128,64,64]
        x = self.relu3_1(x) 
        x = self.conv4(x)   # [?,256,32,32]
        if self.activations:
            x = self.sig(x)
        else:
            x = self.relu4_1(x)
        x = x.view(x.size(0), -1)   # [?, size dependent on image size]
        x = self.embedding(x)       # [?, 512]
        cluster = self.clust(x)
        extra_out = cluster
        clustering_out = self.clustering(cluster) # [?, num_clusters]
        x = self.deembedding(x)             # [?, 262144]
        x = self.relu4_2(x)
        x = x.view(x.size(0), self.filters[3], ((self.input_shape[1]//2//2//2) // 2), ((self.input_shape[1]//2//2//2) // 2)) # [?,256.32,32]
        x = self.deconv4(x) # [?,128,64,64]
        x = self.relu3_2(x)
        x = self.deconv3(x) # [?,64,128,128]
        x = self.relu2_2(x)
        x = self.deconv2(x)  # [?,32,256,256]
        x = self.relu1_2(x)
        x = self.deconv1(x)  # [?;3,512,512]
        if self.activations:
            x = self.tanh(x)
        return x, clustering_out, extra_out

# Convolutional autoencoder with 4 convolutional blocks (BN version)
class TNBC_CAE_bn4(nn.Module):
    def __init__(self, input_shape=[3,128,128], z_dim=512, num_clusters=30, filters=[32, 64, 128, 256], leaky=True, neg_slope=0.01, activations=False, bias=True):
        super(TNBC_CAE_bn4, self).__init__()
        self.activations = activations
        self.pretrained = False
        self.num_clusters = num_clusters
        self.input_shape = input_shape
        self.filters = filters
        self.z_dim = z_dim
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)

        self.conv1 = nn.Conv2d(input_shape[0], filters[0], 5, stride=2, padding=2, bias=bias)
        self.bn1_1 = nn.BatchNorm2d(filters[0])
        self.conv2 = nn.Conv2d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
        self.bn2_1 = nn.BatchNorm2d(filters[1])
        self.conv3 = nn.Conv2d(filters[1], filters[2], 5, stride=2, padding=2, bias=bias)
        self.bn3_1 = nn.BatchNorm2d(filters[2])
        self.conv4 = nn.Conv2d(filters[2], filters[3], 3, stride=2, padding=1, bias=bias)

        lin_features_len = ((input_shape[1] // 2 // 2 // 2) // 2) * ((input_shape[1] // 2 // 2 // 2) // 2) * \
                           filters[3]
        self.embedding = nn.Linear(lin_features_len, z_dim, bias=bias)
        self.deembedding = nn.Linear(z_dim, lin_features_len, bias=bias)
        out_pad = 1 if input_shape[1] // 2 // 2 // 2 % 2 == 0 else 0
        self.deconv4 = nn.ConvTranspose2d(filters[3], filters[2], 3, stride=2, padding=1, output_padding=out_pad,
                                          bias=bias)
        self.bn4_2 = nn.BatchNorm2d(filters[2])
        out_pad = 1 if input_shape[1] // 2 // 2 % 2 == 0 else 0
        self.deconv3 = nn.ConvTranspose2d(filters[2], filters[1], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.bn3_2 = nn.BatchNorm2d(filters[1])
        out_pad = 1 if input_shape[1] // 2 % 2 == 0 else 0
        self.deconv2 = nn.ConvTranspose2d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.bn2_2 = nn.BatchNorm2d(filters[0])
        out_pad = 1 if input_shape[1] % 2 == 0 else 0
        self.deconv1 = nn.ConvTranspose2d(filters[0], input_shape[0], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.clustering = ClusterlingLayer(z_dim, num_clusters)
        # ReLU copies for graph representation in tensorboard
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu4_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.relu4_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.bn1_1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.bn2_1(x)
        x = self.conv3(x)
        x = self.relu3_1(x)
        x = self.bn3_1(x)
        x = self.conv4(x)
        if self.activations:
            x = self.sig(x)
        else:
            x = self.relu4_1(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        extra_out = x
        clustering_out = self.clustering(x)
        x = self.deembedding(x)
        x = self.relu4_2(x)
        x = x.view(x.size(0), self.filters[3], ((self.input_shape[1]//2//2//2) // 2), ((self.input_shape[1]//2//2//2) // 2))
        x = self.deconv4(x)
        x = self.relu3_2(x)
        x = self.bn4_2(x)
        x = self.deconv3(x)
        x = self.relu2_2(x)
        x = self.bn3_2(x)
        x = self.deconv2(x)
        x = self.relu1_2(x)
        x = self.bn2_2(x)
        x = self.deconv1(x)
        x = self.sig(x)
        return x, clustering_out, extra_out

# Convolutional autoencoder with 5 convolutional blocks
class TNBC_CAE_5(nn.Module):
    def __init__(self, input_shape=[3,128,128], z_dim=512, num_clusters=30, filters=[32, 64, 128, 256, 512], leaky=True, neg_slope=0.01, activations=False, bias=True):
        super(TNBC_CAE_5, self).__init__()
        self.activations = activations
        self.pretrained = False
        self.num_clusters = num_clusters
        self.input_shape = input_shape
        self.filters = filters
        self.relu = nn.ReLU(inplace=False)
        self.z_dim = z_dim

        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)

        self.conv1 = nn.Conv2d(input_shape[0], filters[0], 5, stride=2, padding=2, bias=bias)
        self.conv2 = nn.Conv2d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
        self.conv3 = nn.Conv2d(filters[1], filters[2], 5, stride=2, padding=2, bias=bias)
        self.conv4 = nn.Conv2d(filters[2], filters[3], 3, stride=2, padding=1, bias=bias)
        self.conv5 = nn.Conv2d(filters[3], filters[4], 3, stride=1, padding=1, bias=bias)

        lin_features_len = ((input_shape[1] // 2 // 2) // 2) * (
                    (input_shape[1] // 2 // 2) // 2 // 2) * filters[3]
        self.embedding = nn.Linear(lin_features_len, z_dim, bias=bias)
        
        self.deembedding = nn.Linear(z_dim, lin_features_len, bias=bias)
        out_pad = 1 if input_shape[1] // 2 // 2 // 2 // 2 % 2 == 0 else 0
        self.deconv5 = nn.ConvTranspose2d(filters[4], filters[3], 3, stride=1, padding=1, output_padding=0,
                                          bias=bias)
        out_pad = 1 if input_shape[1] // 2 // 2 // 2 % 2 == 0 else 0
        self.deconv4 = nn.ConvTranspose2d(filters[3], filters[2], 3, stride=2, padding=1, output_padding=out_pad,
                                          bias=bias)
        out_pad = 1 if input_shape[1] // 2 // 2 % 2 == 0 else 0
        self.deconv3 = nn.ConvTranspose2d(filters[2], filters[1], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        out_pad = 1 if input_shape[1] // 2 % 2 == 0 else 0
        self.deconv2 = nn.ConvTranspose2d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        out_pad = 1 if input_shape[1] % 2 == 0 else 0
        self.deconv1 = nn.ConvTranspose2d(filters[0], input_shape[0], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.clustering = ClusterlingLayer(z_dim, num_clusters)
        # ReLU copies for graph representation in tensorboard
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu4_1 = copy.deepcopy(self.relu)
        self.relu5_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.relu4_2 = copy.deepcopy(self.relu)
        self.relu5_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.conv3(x)
        x = self.relu3_1(x)
        x = self.conv4(x)
        x = self.relu4_1(x)
        x = self.conv5(x)
        if self.activations:
            x = self.sig(x)
        else:
            x = self.relu5_1(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        extra_out = x
        clustering_out = self.clustering(x)
        x = self.deembedding(x)
        x = self.relu4_2(x)
        x = x.view(x.size(0), self.filters[4], ((self.input_shape[1]//2//2//2) // 2), ((self.input_shape[1]//2//2//2) // 2))
        x = self.deconv5(x)
        x = self.relu4_2(x)
        x = self.deconv4(x)
        x = self.relu3_2(x)
        x = self.deconv3(x)
        x = self.relu2_2(x)
        x = self.deconv2(x)
        x = self.relu1_2(x)
        x = self.deconv1(x)
        x = self.sig(x)
        return x, clustering_out, extra_out            

# Convolutional autoencoder with 5 convolutional blocks (BN version)
class TNBC_CAE_bn5(nn.Module):
    def __init__(self, input_shape=[3,128,128], z_dim=512, num_clusters=30, filters=[32, 64, 128, 256, 512], leaky=True, neg_slope=0.01, activations=False, bias=True):
        super(TNBC_CAE_bn5, self).__init__()
        self.activations = activations
        self.pretrained = False
        self.num_clusters = num_clusters
        self.input_shape = input_shape
        self.filters = filters
        self.relu = nn.ReLU(inplace=False)
        self.z_dim = z_dim

        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)

        self.conv1 = nn.Conv2d(input_shape[0], filters[0], 5, stride=2, padding=2, bias=bias)
        self.bn1_1 = nn.BatchNorm2d(filters[0])
        self.conv2 = nn.Conv2d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
        self.bn2_1 = nn.BatchNorm2d(filters[1])
        self.conv3 = nn.Conv2d(filters[1], filters[2], 5, stride=2, padding=2, bias=bias)
        self.bn3_1 = nn.BatchNorm2d(filters[2])
        self.conv4 = nn.Conv2d(filters[2], filters[3], 5, stride=2, padding=2, bias=bias)
        self.bn4_1 = nn.BatchNorm2d(filters[3])
        self.conv5 = nn.Conv2d(filters[3], filters[4], 3, stride=2, padding=1, bias=bias)

        lin_features_len = ((input_shape[1] // 2 // 2 // 2 // 2) // 2) * (
                    (input_shape[1] // 2 // 2 // 2 // 2) // 2) * filters[4]
        self.embedding = nn.Linear(lin_features_len, z_dim, bias=bias)

        self.deembedding = nn.Linear(z_dim, lin_features_len, bias=bias)
        out_pad = 1 if input_shape[1] // 2 // 2 // 2 // 2 % 2 == 0 else 0
        self.deconv5 = nn.ConvTranspose2d(filters[4], filters[3], 3, stride=2, padding=1, output_padding=out_pad,
                                          bias=bias)
        self.bn5_2 = nn.BatchNorm2d(filters[3])
        out_pad = 1 if input_shape[1] // 2 // 2 // 2 % 2 == 0 else 0
        self.deconv4 = nn.ConvTranspose2d(filters[3], filters[2], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.bn4_2 = nn.BatchNorm2d(filters[2])
        out_pad = 1 if input_shape[1] // 2 // 2 % 2 == 0 else 0
        self.deconv3 = nn.ConvTranspose2d(filters[2], filters[1], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.bn3_2 = nn.BatchNorm2d(filters[1])
        out_pad = 1 if input_shape[1] // 2 % 2 == 0 else 0
        self.deconv2 = nn.ConvTranspose2d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.bn2_2 = nn.BatchNorm2d(filters[0])
        out_pad = 1 if input_shape[1] % 2 == 0 else 0
        self.deconv1 = nn.ConvTranspose2d(filters[0], input_shape[0], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.clustering = ClusterlingLayer(z_dim, num_clusters)
        # ReLU copies for graph representation in tensorboard
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu4_1 = copy.deepcopy(self.relu)
        self.relu5_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.relu4_2 = copy.deepcopy(self.relu)
        self.relu5_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.bn1_1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.bn2_1(x)
        x = self.conv3(x)
        x = self.relu3_1(x)
        x = self.bn3_1(x)
        x = self.conv4(x)
        x = self.relu4_1(x)
        x = self.bn4_1(x)
        x = self.conv5(x)
        if self.activations:
            x = self.sig(x)
        else:
            x = self.relu5_1(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        extra_out = x
        clustering_out = self.clustering(x)
        x = self.deembedding(x)
        x = self.relu5_2(x)
        x = x.view(x.size(0), self.filters[4], ((self.input_shape[1]//2//2//2//2) // 2), ((self.input_shape[1]//2//2//2//2) // 2))
        x = self.deconv5(x)
        x = self.relu4_2(x)
        x = self.bn5_2(x)
        x = self.deconv4(x)
        x = self.relu3_2(x)
        x = self.bn4_2(x)
        x = self.deconv3(x)
        x = self.relu2_2(x)
        x = self.bn3_2(x)
        x = self.deconv2(x)
        x = self.relu1_2(x)
        x = self.bn2_2(x)
        x = self.deconv1(x)
        x = self.sig(x)
        return x, clustering_out, extra_out

class ClusterlingLayer(nn.Module):
    def __init__(self, in_features=10, out_features=10, alpha=1.0):
        super(ClusterlingLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight = nn.init.xavier_uniform_(self.weight)

    def forward(self, x): # x.shape=[?,z_dim]
        x = x.unsqueeze(1) - self.weight # [?, n_clust, z_dim]
        x = torch.mul(x, x)              # [?, n_clust, z_dim]
        x = torch.sum(x, dim=2)          # [?, n_clust]
        x = 1.0 + (x / self.alpha)       # [?, n_clust]
        x = 1.0 / x                      # [?, n_clust]
        x = x ** ((self.alpha +1.0) / 2.0)
        x = torch.t(x) / torch.sum(x, dim=1)
        x = torch.t(x)
        return x

    def set_weight(self, tensor):
        self.weight = nn.Parameter(tensor)

# Still VAE. Decoders is class px
class px(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(px, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(z_dim, 1024, bias=False), nn.BatchNorm1d(1024), nn.ReLU())
        self.up1 = nn.Upsample(8)
        self.de1 = nn.Sequential(nn.ConvTranspose2d(64, 128, kernel_size=5, stride=1, padding=0, bias=False), nn.BatchNorm2d(128), nn.ReLU())
        self.up2 = nn.Upsample(24)
        self.de2 = nn.Sequential(nn.ConvTranspose2d(128, 256, kernel_size=5, stride=1, padding=0, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self.de3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1))

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.de1[0].weight)
        torch.nn.init.xavier_uniform_(self.de2[0].weight)
        torch.nn.init.xavier_uniform_(self.de3[0].weight)
        self.de3[0].bias.data.zero_()

    def forward(self, z):
        h = self.fc1(z)
        h = h.view(-1, 64, 4, 4)
        h = self.up1(h)
        h = self.de1(h)
        h = self.up2(h)
        h = self.de2(h)
        loc_img = self.de3(h)

        return loc_img

# Still VAE. Encoders is class qz
class qz(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(qz, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
        )

        self.fc11 = nn.Sequential(nn.Linear(1024, z_dim))
        self.fc12 = nn.Sequential(nn.Linear(1024, z_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.encoder[0].weight)
        torch.nn.init.xavier_uniform_(self.encoder[4].weight)
        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(-1, 1024)
        zd_loc = self.fc11(h)
        zd_scale = self.fc12(h) + 1e-7

        return zd_loc, zd_scale

# VAE2 is from Jeppe's implementation
class VAE2(nn.Module):
    def __init__(self, args):
        super(VAE2, self).__init__()
        self.z_dim = args.z_dim
        #self.d_dim = args.d_dim
        self.x_dim = args.custom_img_size[0]
        
        self.px = px(self.x_dim, self.z_dim)

        self.qz = qz(self.x_dim, self.z_dim)

        self.beta = args.beta

        self.cuda()
    def forward(self, x):
        # Encode
        z_q_loc, z_q_scale = self.qz(x)

        # Reparameterization trick
        qz = dist.Normal(z_q_loc, z_q_scale)
        z_q = qz.rsample()

        # Decode
        x_recon = self.px(z_q)

        # Priors
        z_p_loc, z_p_scale = torch.zeros(z_q.size()[0], self.z_dim).cuda(),\
                               torch.ones(z_q.size()[0], self.z_dim).cuda()
        pz = dist.Normal(z_p_loc, z_p_scale)

        return x_recon, qz, pz, z_q

    def loss_function(self, x):
        x_recon, qz, pz, z_q = self.forward(x)

        x_recon = x_recon.view(-1, 256)
        x_target = (x.view(-1) * 255).long()
        CE_x = F.cross_entropy(x_recon, x_target, reduction='sum')

        KL_z = torch.sum(pz.log_prob(z_q) - qz.log_prob(z_q))

        return CE_x - self.beta * KL_z

class VaDE(nn.Module):
    def __init__(self, input_dim=784, z_dim=20, n_centroids=10, binary=True,
        encodeLayer=[500,500,2000], decodeLayer=[2000,500,500]):
        super(self.__class__, self).__init__()
        self.z_dim = z_dim
        self.n_centroids = n_centroids
        self.encoder = self.buildNetwork([input_dim] + encodeLayer)
        self.decoder = self.buildNetwork([z_dim] + decodeLayer)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._enc_log_sigma = nn.Linear(encodeLayer[-1], z_dim)
        self._dec = nn.Linear(decodeLayer[-1], input_dim)
        self._dec_act = None
        if binary:
            self._dec_act = nn.Sigmoid()

        self.create_gmmparam(n_centroids, z_dim)
    
    def forward(self, x):
        h = self.encoder(x)
        mu = self._enc_mu(h)
        logvar = self._enc_log_sigma(h)
        z = self.reparameterize(mu, logvar)
        return z, self.decode(z), mu, logvar

    def create_gmmparam(self, n_centroids, z_dim):
        self.theta_p = nn.Parameter(torch.ones(n_centroids)/n_centroids)
        self.u_p = nn.Parameter(torch.zeros(z_dim, n_centroids))
        self.lambda_p = nn.Parameter(torch.ones(z_dim, n_centroids))

    def initialize_gmm(self, dataloader):
        self.eval()
        data = []
        for batch_idx, (inputs, _) in enumerate(dataloader):
            inputs = inputs.view(inputs.size(0), -1).float()
            inputs = Variable(inputs)
            z, outputs, mu, logvar = self.forward(inputs)
            data.append(z.data.cpu().numpy())
        data = np.concatenate(data)
        gmm = GaussianMixture(n_components=self.n_centroids,covariance_type='diag')
        gmm.fit(data)
        self.u_p.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
        self.lambda_p.data.copy_(torch.from_numpy(gmm.covariances_.T.astype(np.float32)))

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          # num = np.array([[ 1.096506  ,  0.3686553 , -0.43172026,  1.27677995,  1.26733758,
          #       1.30626082,  0.14179629,  0.58619505, -0.76423112,  2.67965817]], dtype=np.float32)
          # num = np.repeat(num, mu.size()[0], axis=0)
          # eps = Variable(torch.from_numpy(num))
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, z):
        h = self.decoder(z)
        x = self._dec(h)
        if self._dec_act is not None:
            x = self._dec_act(x)
        return x

    def get_gamma(self, z, z_mean, z_log_var):  # predict cluster label out of n clusters
        Z = z.unsqueeze(2).expand(z.size()[0], z.size()[1], self.n_centroids) # NxDxK
        z_mean_t = z_mean.unsqueeze(2).expand(z_mean.size()[0], z_mean.size()[1], self.n_centroids)
        z_log_var_t = z_log_var.unsqueeze(2).expand(z_log_var.size()[0], z_log_var.size()[1], self.n_centroids)
        u_tensor3 = self.u_p.unsqueeze(0).expand(z.size()[0], self.u_p.size()[0], self.u_p.size()[1]) # NxDxK
        lambda_tensor3 = self.lambda_p.unsqueeze(0).expand(z.size()[0], self.lambda_p.size()[0], self.lambda_p.size()[1])
        theta_tensor2 = self.theta_p.unsqueeze(0).expand(z.size()[0], self.n_centroids) # NxK

        p_c_z = torch.exp(torch.log(theta_tensor2) - torch.sum(0.5*torch.log(2*math.pi*lambda_tensor3)+\
            (Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)) + 1e-10 # NxK
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

        return gamma

    def loss_function(self, recon_x, x, z, z_mean, z_log_var):
        Z = z.unsqueeze(2).expand(z.size()[0], z.size()[1], self.n_centroids) # NxDxK
        z_mean_t = z_mean.unsqueeze(2).expand(z_mean.size()[0], z_mean.size()[1], self.n_centroids)
        z_log_var_t = z_log_var.unsqueeze(2).expand(z_log_var.size()[0], z_log_var.size()[1], self.n_centroids)
        u_tensor3 = self.u_p.unsqueeze(0).expand(z.size()[0], self.u_p.size()[0], self.u_p.size()[1]) # NxDxK
        lambda_tensor3 = self.lambda_p.unsqueeze(0).expand(z.size()[0], self.lambda_p.size()[0], self.lambda_p.size()[1])
        theta_tensor2 = self.theta_p.unsqueeze(0).expand(z.size()[0], self.n_centroids) # NxK
        
        p_c_z = torch.exp(torch.log(theta_tensor2) - torch.sum(0.5*torch.log(2*math.pi*lambda_tensor3)+\
            (Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)) + 1e-10 # NxK
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True) # NxK
        
        BCE = -torch.sum(x*torch.log(torch.clamp(recon_x, min=1e-10))+
            (1-x)*torch.log(torch.clamp(1-recon_x, min=1e-10)), 1)
        logpzc = torch.sum(0.5*gamma*torch.sum(math.log(2*math.pi)+torch.log(lambda_tensor3)+\
            torch.exp(z_log_var_t)/lambda_tensor3 + (z_mean_t-u_tensor3)**2/lambda_tensor3, dim=1), dim=1)
        qentropy = -0.5*torch.sum(1+z_log_var+math.log(2*math.pi), 1)
        logpc = -torch.sum(torch.log(theta_tensor2)*gamma, 1)
        logqcx = torch.sum(torch.log(gamma)*gamma, 1)

        # Normalise by same number of elements as in reconstruction
        loss = torch.mean(BCE + logpzc + qentropy + logpc + logqcx)

        return loss
    
    def buildNetwork(self, layers, activation="relu", dropout=0):
        net = []
        for i in range(1, len(layers)):
            net.append(nn.Linear(layers[i-1], layers[i]))
            if activation=="relu":
                net.append(nn.ReLU())
            elif activation=="sigmoid":
                net.append(nn.Sigmoid())
            if dropout > 0:
                net.append(nn.Dropout(dropout))
        return nn.Sequential(*net)

    def cluster_acc(self, Y_pred, Y):
        assert Y_pred.size == Y.size
        D = max(Y_pred.max(), Y.max())+1
        w = np.zeros((D,D), dtype=np.int64)
        for i in range(Y_pred.size):
            w[Y_pred[i], Y[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w

class resnet18(nn.Module):
    def __init__(self, input_shape = [3,128,128], z_dim=512, h_dim=2048): # change inter_dims to 512, h_dim to 2048
        super(resnet18,self).__init__()
        nFilters = 64
        resnet = nmodels.resnet18(pretrained=True)
        output_channels = 3
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.input_shape = input_shape
        modules = list(resnet.children())[:-2]      # delete the last fc layer + AvgPoolLayer.
        self.encoder = nn.Sequential(
            *modules,
            nn.AdaptiveAvgPool2d(1),)
        lin_features_len = ((self.input_shape[1] // 2 // 2 // 2) // 2) * ((self.input_shape[1] // 2 // 2 // 2) // 2) * 512
        self.dfc2 = nn.Linear(z_dim, lin_features_len)
        # self.bn2 = nn.BatchNorm1d(lin_features_len)
        self.relu2 = nn.ReLU(True)

        net = []
        net.append(nn.ConvTranspose2d(z_dim, 4*nFilters, kernel_size=3, padding=1))
        net.append(nn.BatchNorm2d(4*nFilters))
        net.append(nn.ReLU(True))
        # net.append(nn.ConvTranspose2d(4*nFilters, 4*nFilters, kernel_size=3, stride=2, padding=1))
        # net.append(nn.BatchNorm2d(4*nFilters))
        # net.append(nn.ReLU(True))
        net.append(nn.ConvTranspose2d(4*nFilters, 2*nFilters, kernel_size=5, stride=2, padding=2, output_padding=1))
        net.append(nn.BatchNorm2d(2*nFilters))
        net.append(nn.ReLU(True))
        net.append(nn.ConvTranspose2d(2*nFilters, 2*nFilters, kernel_size=5, stride=2, padding=2, output_padding=1))
        net.append(nn.BatchNorm2d(2*nFilters))
        net.append(nn.ReLU(True))
        net.append(nn.ConvTranspose2d(2*nFilters, nFilters, kernel_size=5, stride=2, padding=2, output_padding=1))
        net.append(nn.BatchNorm2d(nFilters))
        net.append(nn.ReLU(True))
        net.append(nn.ConvTranspose2d(nFilters, output_channels, kernel_size=5, stride=2, padding=2, output_padding=1))
        net.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*net)
        
        # net = []
        # net.append(nn.Upsample(8))
        # net.append(nn.Conv2d(z_dim, 4*nFilters, 3, stride=1, padding=1))
        # net.append(nn.ReLU(True))

        # net.append(nn.Upsample(16))
        # net.append(nn.Conv2d(4*nFilters,2*nFilters,3, stride=1, padding=1))
        # net.append(nn.ReLU(True))
        
        # net.append(nn.Upsample(32))
        # net.append(nn.Conv2d(2*nFilters,3,3, stride=1, padding=1))
        # net.append(nn.ReLU(True))

        # net.append(nn.Upsample(64))
        # net.append(nn.Conv2d(nFilters,32,5, stride=1, padding=1))
        # net.append(nn.ReLU(True))
        
        # net.append(nn.Upsample(128))
        # net.append(nn.Conv2d(32, 3, 3, stride=1, padding=1))
        # net.append(nn.Sigmoid())
        # self.decoder = nn.Sequential(*net)

    
    def forward(self,x):    #[?,3,128,128]
        x = self.encoder(x) #[?,512,1,1]
        x = x.view(-1, 512)
        z = x
        x = self.dfc2(x)    #[?,h_dim]
        # x = self.bn2(x)
        x = self.relu2(x)
        # x = x.view(-1, self.h_dim,1,1)  #[?,2048,1,1]
        # x_tilde = self.decoder(x)       #[?,3,128,128]
        # x_tilde = x.view(x.size(0), -1)
        x = x.view(-1,self.z_dim,8,8)
        x_tilde = self.decoder(x)
        return x_tilde, z

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
    
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class res_encoder(nn.Module):
    def __init__(self, block, layers, z_dim):
        super(res_encoder, self).__init__()
        self.inplanes = 64
        self.z_dim = z_dim 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, self.z_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class res_decoder(nn.Module):
    def __init__(self, zsize=2048):
        super(res_decoder,self).__init__()
        self.dfc3 = nn.Linear(zsize, 4096)
        self.bn3 = nn.BatchNorm1d(4096)
        self.dfc2 = nn.Linear(4096, 4096)   
        self.bn2 = nn.BatchNorm1d(4096)
        self.dfc1 = nn.Linear(4096,256 * 8 * 8)
        self.bn1 = nn.BatchNorm1d(256*8*8)
        self.upsample1=nn.Upsample(scale_factor=2)
        self.dconv5 = nn.ConvTranspose2d(256, 256, 3, padding = 1)
        self.dconv4 = nn.ConvTranspose2d(256, 384, 3, stride=2, padding = 1, output_padding=1)
        self.dconv3 = nn.ConvTranspose2d(384, 192, 3, padding = 1)
        self.dconv2 = nn.ConvTranspose2d(192, 64, 5, padding = 2)
        self.dconv1 = nn.ConvTranspose2d(64, 3, 7, stride = 1, padding = 3)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        x = self.dfc3(x) 
        x = F.relu(self.bn3(x))
        x = x.view(-1, 4096)
        x = self.dfc2(x)
        x = F.relu(self.bn2(x))
        x = self.dfc1(x)
        x = F.relu(self.bn1(x))
        x = x.view(-1,256,8,8)
        x = self.upsample1(x)
        x = self.dconv5(x)
        x = F.relu(x)
        x = F.relu(self.dconv4(x))
        x = F.relu(self.dconv3(x))
        x = self.upsample1(x)
        x = self.dconv2(x)
        x = F.relu(x)
        
        x = self.upsample1(x)
        x = self.dconv1(x)
        x = self.sig(x)
        return x

class res_AE(nn.Module):
    def __init__(self, num_clusters, z_dim):
        super(res_AE, self).__init__()
        self.z_dim = z_dim
        self.encoder = res_encoder(Bottleneck, [3, 4, 6, 3], z_dim)
        self.decoder = res_decoder()

    def forward(self, x):
        x = self.encoder(x)
        z = x
        x = self.decoder(x)
        return x, z
