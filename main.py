# Thomas' main Triple negative breast cancer project, Aug 27th 2019
import os
os.chdir('C:\Source\Research Repositories\TNBC')
import numpy as np
import argparse
from torchvision.datasets import MNIST
from tqdm.autonotebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
import time
import torch
import fnmatch
from torch import nn, optim
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import torchvision.transforms as transforms
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image #
from torch.optim import lr_scheduler 

import mods
import datagen
import utils
import trainers

# Defining hyperparameters
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
parser = argparse.ArgumentParser(description='This is DCEC model with clustering')
parser.add_argument('--net_architecture', default='TNBC_CAE', choices=['MnistCAE', 'TNBC_CAE', 'CAE_3', 'CAE_bn3', 'VAE2', 'VaDE'], 
                    help='network architecture used. Must match the classes from mods.py')
parser.add_argument('--pretrain', default=True, type=str2bool, help='perform autoencoder pretraining')
parser.add_argument('--mode', default='train_full', choices=['train_full', 'pretrain'], help='mode')
parser.add_argument('--dataset', default='TNBC',
                    choices=['MNIST-train', 'TNBC'],
                    help='custom or prepared dataset')

parser.add_argument('--epochs', default=1000, type=int, help='clustering epochs')
parser.add_argument('--epochs_pretrain', default=100, type=int, help='pretraining epochs')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--num_clusters', default=10, type=int, help='number of clusters')

# VAE parameters
parser.add_argument('--z_dim', default=20, type=int, help='z-dim for VAE')
parser.add_argument('--beta', default=1, type=float, help='Beta value for VAE')

parser.add_argument('--tensorboard', default=True, type=bool, help='export training stats to tensorboard')
parser.add_argument('--pretrained_net', default=2, help='index or path of pretrained net')
parser.add_argument('--dataset_path', default=r"C:\Source\Research Repositories\TNBC\data\dataframe_512.csv", help='path to dataset')
parser.add_argument('--rate', default=0.001, type=float, help='learning rate for clustering')
parser.add_argument('--rate_pretrain', default=0.001, type=float, help='learning rate for pretraining')
parser.add_argument('--weight', default=0.0, type=float, help='weight decay for clustering')
parser.add_argument('--weight_pretrain', default=0.0, type=float, help='weight decay for clustering')
parser.add_argument('--sched_step', default=200, type=int, help='scheduler steps for rate update')
parser.add_argument('--sched_step_pretrain', default=200, type=int,
                    help='scheduler steps for rate update - pretrain')
parser.add_argument('--sched_gamma', default=0.1, type=float, help='scheduler gamma for rate update')
parser.add_argument('--sched_gamma_pretrain', default=0.1, type=float,
                    help='scheduler gamma for rate update - pretrain')
parser.add_argument('--printing_frequency', default=50, type=int, help='training stats printing frequency')
parser.add_argument('--gamma', default=0.1, type=float, help='clustering loss weight')
parser.add_argument('--update_interval', default=80, type=int, help='update interval for target distribution')
parser.add_argument('--tol', default=1e-2, type=float, help='stop criterium tolerance')
parser.add_argument('--custom_img_size', default=[3, 512, 512], nargs=3, type=int, help='size of custom images')
parser.add_argument('--leaky', default=True, type=str2bool)
parser.add_argument('--neg_slope', default=0.01, type=float)
parser.add_argument('--activations', default=False, type=str2bool)
parser.add_argument('--bias', default=True, type=str2bool)
args = parser.parse_args()
print(args)

if args.mode == 'pretrain' and not args.pretrain:
    print("Nothing to do :(")
    exit()

board = args.tensorboard

# Deal with pretraining option and way of showing network path
# if args.pretrain=False params['pretrained']=False
pretrain = args.pretrain
net_is_path = True
if not pretrain:
    try:
        int(args.pretrained_net)
        idx = args.pretrained_net
        net_is_path = False
    except:
        pass
params = {'pretrain': pretrain}

# Directories
# Create directories structure
dirs = ['runs', 'reports', 'nets']
list(map(lambda x: os.makedirs(x, exist_ok=True), dirs))

# Net architecture (model name) 
model_name = args.net_architecture

# Indexing (for automated reports saving) - allows to run many trainings and get all the reports collected.
# If model name is already taken, a new idx number is added to new model for saving and logging.
if pretrain or (not pretrain and net_is_path):
    reports_list = sorted(os.listdir('reports'), reverse=True)
    if reports_list:
        for file in reports_list:
            # print(file)
            if fnmatch.fnmatch(file, model_name + '*'):
                idx = int(str(file)[-7:-4]) + 1
                break
    try:
        idx
    except NameError:
        idx = 1

# Base filename
name = model_name + '_' + str(idx).zfill(3)

# Filenames for report, weights and regenerated pictures
name_txt = name + '.txt'
name_net = name
pretrained = name + '_pretrained.pt'

# Arrange filenames for report, network weights, pretrained network weights
name_txt = os.path.join('reports', name_txt)
name_net = os.path.join('nets', name_net)
name_pic = os.path.join('runs', name)
if net_is_path and not pretrain:
    pretrained = args.pretrained_net
else:
    pretrained = os.path.join('nets', pretrained)
if not pretrain and not os.path.isfile(pretrained):
    print("No pretrained weights, try again choosing pretrained network or create new with pretrain=True")

model_files = [name_net, pretrained]
params['model_files'] = model_files
params['pic_path'] = name_pic
# Open file
if pretrain:
    f = open(name_txt, 'w')
else:
    f = open(name_txt, 'a')
params['txt_file'] = f

# Delete tensorboard entry if exist (not to overlap as the charts become unreadable)
try:
    os.system("rm -rf runs/" + name)
except:
    pass

# Initialize tensorboard writer
if board:
    writer = SummaryWriter('runs/' + name)
    params['writer'] = writer
else:
    params['writer'] = None


# Hyperparameters

# Used dataset
dataset = args.dataset

# Batch size
batch = args.batch_size
params['batch'] = batch
# Number of workers (typically 4*num_of_GPUs)
workers = 4
# Learning rate
rate = args.rate
rate_pretrain = args.rate_pretrain
# Adam params
# Weight decay
weight = args.weight
weight_pretrain = args.weight_pretrain
# Scheduler steps for rate update
sched_step = args.sched_step
sched_step_pretrain = args.sched_step_pretrain
# Scheduler gamma - multiplier for learning rate
sched_gamma = args.sched_gamma
sched_gamma_pretrain = args.sched_gamma_pretrain

# Number of epochs
epochs = args.epochs
pretrain_epochs = args.epochs_pretrain
params['pretrain_epochs'] = pretrain_epochs

# Printing frequency
print_freq = args.printing_frequency
params['print_freq'] = print_freq

# Clustering loss weight:
gamma = args.gamma
params['gamma'] = gamma

# Update interval for target distribution:
update_interval = args.update_interval
params['update_interval'] = update_interval

# Tolerance for label changes:
tol = args.tol
params['tol'] = tol

# Number of clusters
num_clusters = args.num_clusters

# Loading data from datagenerator
img_size = args.custom_img_size
train_batch_size = val_batch_size = batch
if dataset == 'MNIST-train':
    train_loader, val_loader = datagen.get_data_loaders(dataset, train_batch_size, val_batch_size) # TO DO: Add dataloader for TNBC
if dataset == 'TNBC':
    train_transform = val_transform = transforms.Compose([transforms.ToTensor()])
    trainer = datagen.DataGen_TNBC('train', args.dataset_path, transform = train_transform)
    train_loader = data.DataLoader(trainer, batch_size=train_batch_size, shuffle=False)
    val = datagen.DataGen_TNBC('eval', args.dataset_path, transform = val_transform)
    val_loader = data.DataLoader(val, batch_size=val_batch_size, shuffle=False)

dataset_size = len(train_loader.dataset)
params['dataset_size'] = dataset_size

# GPU check
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tmp = "\nPerforming calculations on:\t" + str(device)
utils.print_both(f, tmp + '\n')
params['device'] = device


# Report for settings
tmp = "Training the '" + model_name + "' architecture"
utils.print_both(f, tmp)
tmp = "\n" + "The following parameters are used:"
utils.print_both(f, tmp)
tmp = "Batch size:\t" + str(batch)
utils.print_both(f, tmp)
tmp = "Number of workers:\t" + str(workers)
utils.print_both(f, tmp)
tmp = "Learning rate:\t" + str(rate)
utils.print_both(f, tmp)
tmp = "Pretraining learning rate:\t" + str(rate_pretrain)
utils.print_both(f, tmp)
tmp = "Weight decay:\t" + str(weight)
utils.print_both(f, tmp)
tmp = "Pretraining weight decay:\t" + str(weight_pretrain)
utils.print_both(f, tmp)
tmp = "Scheduler steps:\t" + str(sched_step)
utils.print_both(f, tmp)
tmp = "Scheduler gamma:\t" + str(sched_gamma)
utils.print_both(f, tmp)
tmp = "Pretraining scheduler steps:\t" + str(sched_step_pretrain)
utils.print_both(f, tmp)
tmp = "Pretraining scheduler gamma:\t" + str(sched_gamma_pretrain)
utils.print_both(f, tmp)
tmp = "Number of epochs of training:\t" + str(epochs)
utils.print_both(f, tmp)
tmp = "Number of epochs of pretraining:\t" + str(pretrain_epochs)
utils.print_both(f, tmp)
tmp = "Clustering loss weight:\t" + str(gamma)
utils.print_both(f, tmp)
tmp = "Update interval for target distribution:\t" + str(update_interval)
utils.print_both(f, tmp)
tmp = "Stop criterium tolerance:\t" + str(tol)
utils.print_both(f, tmp)
tmp = "Number of clusters:\t" + str(num_clusters)
utils.print_both(f, tmp)
tmp = "Leaky relu:\t" + str(args.leaky)
utils.print_both(f, tmp)
tmp = "Leaky slope:\t" + str(args.neg_slope)
utils.print_both(f, tmp)
tmp = "Activations:\t" + str(args.activations)
utils.print_both(f, tmp)
tmp = "Bias:\t" + str(args.bias)
utils.print_both(f, tmp)


if model_name == 'MnistCAE' or model_name == 'TNBC_CAE':
    to_eval = "mods." + model_name + "(img_size)"
elif model_name == 'VAE':
    to_eval = "mods." + model_name  + "(image_size=img_size[0]*img_size[1])"
elif model_name == 'VAE2':
    to_eval = "mods." + model_name + "(args)"
elif model_name == 'VaDE':
    to_eval = 'mods.' + model_name + "(input_dim=img_size[0]*img_size[1])"
else:
    to_eval = "mods." + model_name + "(img_size, num_clusters=num_clusters, leaky = args.leaky, neg_slope = args.neg_slope)"

model = eval(to_eval)
model = model.to(device)

# Setting optimizers. Use reconstruction loss only when optimizing AE (MSELoss). Combine with clustering loss if joint optimization.

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=rate, weight_decay=weight)
optimizer_pretrain = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=rate_pretrain, weight_decay=weight_pretrain)
optimizers = [optimizer, optimizer_pretrain]

scheduler = lr_scheduler.StepLR(optimizer, step_size=sched_step, gamma=sched_gamma)
scheduler_pretrain = lr_scheduler.StepLR(optimizer_pretrain, step_size=sched_step_pretrain, gamma=sched_gamma_pretrain)
schedulers = [scheduler, scheduler_pretrain]

# Evaluate the proper model and optimizers given model_name
if model_name == 'MnistCAE' or model_name == 'TNBC_CAE':
    criteria = nn.MSELoss(size_average=True)
    model = trainers.trainCAE(model, train_loader, val_loader, criteria, optimizers[1], schedulers[1], pretrain_epochs, params)
if model_name == 'VAE':
    model = trainers.trainVAE(model, train_loader, val_loader, optimizers[1], pretrain_epochs, params)
if model_name == 'VAE2':
    model = trainers.trainVAE2(model, train_loader, val_loader, optimizers[1], params, args)
if model_name == 'VaDE':
    model = trainers.trainVaDE(model, train_loader, val_loader, optimizers[1], pretrain_epochs, params, visualize=True)
if model_name == 'CAE_3' or model_name == 'CAE_bn3':
    criterion_1 = nn.MSELoss(size_average=True)     # Reconstruction loss
    criterion_2 = nn.KLDivLoss(size_average=False)  # Clustering loss
    criteria = [criterion_1, criterion_2]
    if args.mode == 'train_full':
        model = trainers.train_model(model, train_loader, criteria, optimizers, schedulers, epochs, params)
    elif args.mode == 'pretrain':
        model = trainers.pretraining(model, train_loader, criteria[0], optimizers[1], schedulers[1], pretrain_epochs, params)


# save model
torch.save(model.state_dict(), name_net + '.pt')

# if model_name != 'MnistCAE':
    # out_arr, label_arr, preds = trainers.calculate_predictions(model, val_loader, params)


hej = 3