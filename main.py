# Thomas' main Triple negative breast cancer project, Aug 27th 2019
import os
import numpy as np
from models.test import ResNet, BasicBlock, resnet18, MnistResNet, calculate_metric, print_scores
from dataloader.datagen import get_data_loaders

from torchvision.datasets import MNIST
from tqdm.autonotebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
import time
import torch
from torch import nn, optim
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
from torchvision.utils import save_image # 

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

for epoch in range(epochs):
    total_loss = 0

    # progress bar (works in Jupyter notebook too!)
    progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)

    # ----------------- TRAINING  -------------------- 
    # set model to training
    model.train()
    
    for i, data in progress:
        X, y = data[0].to(device), data[1].to(device)
        
        # training step for single batch
        model.zero_grad()
        outputs = model(X)
        loss = loss_function(outputs, X)
        loss.backward()
        optimizer.step()

        # getting training quality data
        current_loss = loss.item()
        total_loss += current_loss

        # updating progress bar
        progress.set_description("Loss: {:.4f}".format(total_loss/(i+1)))
        
    # releasing unceseccary memory in GPU
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    
    # ----------------- VALIDATION  ----------------- 
    val_losses = 0
    precision, recall, f1, accuracy = [], [], [], []
    
    # set model to evaluating (testing)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            X, y = data[0].to(device), data[1].to(device)
            outputs = model(X) # this get's the prediction from the network
            val_losses += loss_function(outputs, X)

            
            # calculate P/R/F1/A metrics for batch
            # for acc, metric in zip((precision, recall, f1, accuracy), 
            #                        (precision_score, recall_score, f1_score, accuracy_score)):
            #     acc.append(
            #         calculate_metric(metric, y.cpu(), predicted_classes.cpu())
            #     )
          
    print(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")
    # print_scores(precision, recall, f1, accuracy, val_batches)
    losses.append(total_loss/batches) # for plotting learning curve
    if epoch % 10 == 0:
        save_image(outputs, './TNBC/MNIST/AE/image_{}.png'.format(epoch))
    if epoch == epochs-1:
        z1 = []
        labels = []
        for i, data in enumerate(val_loader):
            X, y = data[0].to(device), data[1].to(device) 
            z = model.embed(X)
            labels.append(y)
            z1.append(z)
        latrep_path = os.path.abspath("C:\Source\Research Repositories\TNBC\MNIST\latrep")
        np.save(os.path.join(latrep_path, 'latrep'), z1)
        np.save(os.path.join(latrep_path, 'labels'), labels)
        torch.save(model.state_dict(), './TNBC/MNIST/weights/weights_{}.pth'.format(epoch))
print(f"Training time: {time.time()-start_ts}s")