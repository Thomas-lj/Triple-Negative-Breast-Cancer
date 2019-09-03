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

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        print(stride, self.inplanes, planes*block.expansion, block.expansion)
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
        #                     self.base_width, previous_dilation, norm_layer))
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            # layers.append(block(self.inplanes, planes, groups=self.groups,
            #                     base_width=self.base_width, dilation=self.dilation,
            #                     norm_layer=norm_layer))
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
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class MnistResNet(ResNet):
    def __init__(self):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), bias=False)
    
        
    def forward(self, x):
        return torch.softmax(
            super(MnistResNet, self).forward(x), dim=-1)
    
    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(self.parameters(), lr=0.02)

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

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

##### ResNet Autoencoder model 

class Encoder(nn.Module):

    def __init__(self, block, layers, num_classes=23):
        self.inplanes = 64
        super (Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#, return_indices = True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, 1000)
	#self.fc = nn.Linear(num_classes,16) 
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

class Decoder(nn.Module):
	def __init__(self):
		super(Decoder,self).__init__()
		self.dfc3 = nn.Linear(zsize, 4096)
		self.bn3 = nn.BatchNorm2d(4096)
		self.dfc2 = nn.Linear(4096, 4096)
		self.bn2 = nn.BatchNorm2d(4096)
		self.dfc1 = nn.Linear(4096,256 * 6 * 6)
		self.bn1 = nn.BatchNorm2d(256*6*6)
		self.upsample1=nn.Upsample(scale_factor=2)
		self.dconv5 = nn.ConvTranspose2d(256, 256, 3, padding = 0)
		self.dconv4 = nn.ConvTranspose2d(256, 384, 3, padding = 1)
		self.dconv3 = nn.ConvTranspose2d(384, 192, 3, padding = 1)
		self.dconv2 = nn.ConvTranspose2d(192, 64, 5, padding = 2)
		self.dconv1 = nn.ConvTranspose2d(64, 3, 12, stride = 4, padding = 4)

	def forward(self,x):#,i1,i2,i3):
		
		x = self.dfc3(x)
		#x = F.relu(x)
		x = F.relu(self.bn3(x))
		
		x = self.dfc2(x)
		x = F.relu(self.bn2(x))
		#x = F.relu(x)
		x = self.dfc1(x)
		x = F.relu(self.bn1(x))
		#x = F.relu(x)
		#print(x.size())
		x = x.view(batch_size,256,6,6)
		#print (x.size())
		x=self.upsample1(x)
		#print x.size()
		x = self.dconv5(x)
		#print x.size()
		x = F.relu(x)
		#print x.size()
		x = F.relu(self.dconv4(x))
		#print x.size()
		x = F.relu(self.dconv3(x))
		#print x.size()		
		x=self.upsample1(x)
		#print x.size()		
		x = self.dconv2(x)
		#print x.size()		
		x = F.relu(x)
		x=self.upsample1(x)
		#print x.size()
		x = self.dconv1(x)
		#print x.size()
		x = F.sigmoid(x)
		#print x
		return x

class Autoencoder(nn.Module):
	def __init__(self):
		super(Autoencoder,self).__init__()
		self.encoder = encoder
		self.binary = Binary()
		self.decoder = Decoder()

	def forward(self,x):
		#x=Encoder(x)
		x = self.encoder(x)
		x = binary.apply(x)
		#print x
		#x,i2,i1 = self.binary(x)
		#x=Variable(x)
		x = self.decoder(x)
		return x

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

class Binary(Function):

    @staticmethod
    def forward(ctx, input):
        return F.relu(Variable(input.sign())).data

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Bottleneck(nn.Module):
    expansion = 1

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
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#, return_indices = True)
        self.layer1 = self._make_layer(self.block, 64, self.layers[0])
        self.layer2 = self._make_layer(self.block, 128, self.layers[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.layers[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.LatRep = nn.AvgPool2d(3, stride=1)
        self.LatRep = nn.AdaptiveAvgPool2d((1,1))
        # self.LatRep = nn.Linear(512 * block.expansion, num_classes)
        # self.LatRep = nn.Linear(1,512)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        ## Decoder part (Hassan Muhammad, fits output size that after dconv0 has x.shape = (batch#, 3, 224, 224))
        # self.dconv4 = nn.ConvTranspose2d(512, 256, 14, padding = 0)
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
        x = self.LatRep(x)
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
        return logits
        # return torch.relu(self.l1(x.view(x.size(0), -1)))

    def loss(self, labels, logits):
        # nll = F.nll_loss(logits, labels)
        nll = F.mse_loss(logits, labels)
        return nll

    def training_step(self, batch, batch_nb): # where the magic happens
        # REQUIRED
        x, y = batch
        x_hat = self.forward(x)
        loss_val = self.loss(x, x_hat)
        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)
        
        output = OrderedDict({'loss': loss_val})
        return output

    def validation_step(self, batch, batch_nb): # where the magic happens
        # OPTIONAL
        x, y = batch
        x_hat = self.forward(x)
        loss_val = self.loss(x, x_hat)
        output = OrderedDict({'val_loss': loss_val})

        return output

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
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

model = CoolSystem(BasicBlock)
# model = MnistResNet()
# binary = Binary()
# encoder = Encoder(BasicBlock, [3, 4, 6, 3])
# decoder = Decoder()
# AE = Autoencoder()
# model.cuda(0)
# model.on_gpu=True

trainer = Trainer(max_nb_epochs=1, overfit_pct=0.1)
# trainer = Trainer(max_nb_epochs=1, overfit_pct=0.1, gpus=[0])    

trainer.fit(model)  



"""
Example template for defining a system
"""
# see pytorch-lightning/examples/new_project_templates/lightning_module_template.py