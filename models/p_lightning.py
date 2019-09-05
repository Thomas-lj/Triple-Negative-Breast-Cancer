# Thomas Aug 28
import os
import torch
import math
import torch.nn as nn
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch import nn
# from test_tube import HyperOptArgumentParser
from collections import OrderedDict
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from pytorch_lightning import Trainer
from torch.autograd import Function

import torchvision.models as models

## Complete ResNet architecture for x classes

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

##### ResNet Autoencoder model 

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class CoolSystem(pl.LightningModule):
    # def __init__(self):
    #     super(CoolSystem, self).__init__()
    #     self.l1 = nn.Linear(28*28, 10)
    def __init__(self, BasicBlock):
        """ pass in arguments from HyperOptArgumentParser (from test_tube import HyperOptArgumentParser) to the model"""
        self.block = BasicBlock
        self.layers = [2, 2, 2, 2]
        super(CoolSystem, self).__init__()
        # self.batch_size = hparams.batch_size
        # self.block = hparams.block
        # self.layers = hparams.layers
        # if you specify an example input, the summary will show input/output for each layer (don't know if I will need it??)
        # self.example_input_array = torch.rand(5, 28 * 28)

        # build model (merge encoder and decoder)
        # self.Encoder()
        # self.Decoder()
        self.__build_model()
        
    def __build_model(self):
        ## Encoder part
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#, return_indices = True)
        self.layer1 = self._make_layer(self.block, 64, self.layers[0])
        self.layer2 = self._make_layer(self.block, 128, self.layers[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.layers[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.layers[3], stride=2)
        self.latrep = nn.AvgPool2d((1,1))
        # self.fc = nn.Linear(512, 1000)
        # self.LatRep = nn.Linear(512, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        ## Decoder part (Hassan Muhammad, fits output size that after dconv0 has x.shape = (batch#, 3, 224, 224))
        # self.dconv4 = nn.ConvTranspose2d(512, 256, 14) # Note, Thomas: if this creates error go back to padding=0 
        # self.dconv3 = nn.ConvTranspose2d(256, 128, 15)
        # self.dconv2 = nn.ConvTranspose2d(128, 64, 29)
        # self.dconv1 = nn.ConvTranspose2d(64, 32, 57)
        # self.dconv0 = nn.ConvTranspose2d(32, 3, 113)
        
        # Decoder part (Thomas' experimenting)
        self.dconv4 = nn.ConvTranspose2d(512, 256, 14)
        self.dconv3 = nn.ConvTranspose2d(256, 1, 15)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
        # if stride != 1 or self.inplanes != planes * block:
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
        # print(x.shape)
        x = self.bn1(x)
        # print(x.shape)
        x = self.relu(x)
        # print(x.shape)
        x = self.maxpool(x)
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)
        x = self.latrep(x)
        LatRep = self.latrep(x)
        # print(x.shape)
        x = self.dconv4(x)
        # print(x.shape)
        x = self.dconv3(x)
        # print(x.shape)
        # x = self.dconv2(x)
        # print(x.shape)
        # x = self.dconv1(x)
        # print(x.shape)
        # x = self.dconv0(x)
        # print(x.shape)
        logits = F.relu(x)
        return logits, LatRep
        
    def loss(self, labels, logits):
        nll = F.mse_loss(logits, labels)
        # nll = F.binary_cross_entropy(logits, labels)
        return nll

    def training_step(self, batch, batch_nb): # where the magic happens
        # REQUIRED
        x, y = batch
        x_hat, LatRep = self.forward(x)
        loss_val = self.loss(x, x_hat)
        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)
        
        output = OrderedDict({'loss': loss_val})
        return output

    def validation_step(self, batch, batch_nb): # where the magic happens
        # OPTIONAL
        x, y = batch
        x_hat, LatRep = self.forward(x)
        loss_val = self.loss(x, x_hat)
        output = OrderedDict({'val_loss': loss_val, 
                                'latent': LatRep,
                                'recon': x_hat})
        return output

    def validation_end(self, outputs): # Note, Thomas: this function seems obsolete for now?? 
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        LatRep = torch.stack([x['latent'] for x in outputs])
        recon = torch.stack([x['recon'] for x in outputs])
        print(LatRep.shape)
        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        # REQUIRED
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def __dataloader(self, train):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (1.0,))])
        dataset = MNIST(os.getcwd(), train=train, transform=transform)
        loader = DataLoader(dataset=dataset, 
            batch_size = batch_size,
            )
        return loader

    @pl.data_loader
    def tng_dataloader(self):
        # REQUIRED
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

# Defining original 
model = CoolSystem(BasicBlock)
# model = MnistResNet()
model_dict = model.state_dict()
# Loading pretrained weights
weight_path = os.path.abspath(r"C:\\Source\\Research Repositories\\TNBC\\models\resnet18_pretrained.pth")
weights = torch.load(weight_path)
# model.load_state_dict(weights)

# Transferring weights from pretrained Resnet to my model only if they match
pretrained_dict = {k: v for k, v in weights.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
# Adjusting input/output layers to match 
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# model.cuda(0)
# model.on_gpu=True

trainer = Trainer(max_nb_epochs=5, overfit_pct=0.02)
# trainer = Trainer(max_nb_epochs=1, overfit_pct=0.1, gpus=[0])    

trained = trainer.fit(model)  
hej = 4



"""
Example template for defining a system
"""
# see pytorch-lightning/examples/new_project_templates/lightning_module_template.py