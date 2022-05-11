# Thomas' main Triple negative breast cancer project, Aug 27th 2019
import os
os.chdir('C:\Source\Research Repositories\TNBC')
import numpy as np
import argparse
from torchvision.datasets import MNIST
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
from torchvision.utils import save_image
from torch.optim import lr_scheduler 

import mods
import datagen
import utils
import trainers

torch.cuda.current_device()

# Defining hyperparameters
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
parser = argparse.ArgumentParser(description='This is DCEC model with clustering')
parser.add_argument('--net_architecture', default='TNBC_CAE_3', choices=['MnistCAE', 'CAE_3', 'TNBC_CAE_3', "TNBC_CAE_4", 'TNBC_CAE_bn4'
                                                                'TNBC_CAE_5', 'TNBC_CAE_bn5', 'TNBC_CAE_bn3', 'VAE2', 'VaDE', 'resnet18'], 
                    help='network architecture used. Must match the classes from mods.py')
parser.add_argument('--pretrain', default=False, type=str2bool, help='True: do pretraining, False: load weight from other pretraind model from pretrained_net_path.')
parser.add_argument('--mode', default='train_full', choices=['train_full', 'pretrain'], help='mode')
parser.add_argument('--dataset', default='Tissues', choices=['MNIST', 'MNIST_subset' 'CIFAR', 'TNBC', 'Tissues', 'Tumor'])

parser.add_argument('--epochs', default=100, type=int, help='clustering epochs')
parser.add_argument('--epochs_pretrain', default=1, type=int, help='pretraining epochs')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--num_clusters', default=4, type=int, help='number of clusters')
parser.add_argument('--z_dim', default=512, type=int, help='z-dim for VAE')

parser.add_argument('--pretrained_net_path', default=r"Z:\Speciale\Results\Results\Vary_K\Tissues\Tissues_TNBC_CAE_3_K4\TissuesTNBC_CAE_3_018_pretrained.pt", help='pretrained weights path')
parser.add_argument('--train_path', default=r"Z:\Speciale\Tissues\Train", help='Choose \mnist_128x128\, \Sampling\ or \Tissues\ ')
parser.add_argument('--eval_path', default=r"Z:\Speciale\Tissues\Eval", help='path to validation dataset')
parser.add_argument('--rate', default=0.0005, type=float, help='learning rate for clustering')
parser.add_argument('--rate_pretrain', default=0.0001, type=float, help='learning rate for pretraining')
parser.add_argument('--weight', default=0.0, type=float, help='weight decay for clustering')
parser.add_argument('--weight_pretrain', default=0.0, type=float, help='weight decay for clustering')
parser.add_argument('--sched_step', default=200, type=int, help='scheduler steps for rate update')
parser.add_argument('--sched_step_pretrain', default=20, type=int,
                    help='scheduler steps for rate update - pretrain')
parser.add_argument('--sched_gamma', default=0.1, type=float, help='scheduler gamma for rate update')
parser.add_argument('--sched_gamma_pretrain', default=0.1, type=float,
                    help='scheduler gamma for rate update - pretrain')
parser.add_argument('--printing_frequency', default=5, type=int, help='training stats printing frequency')
parser.add_argument('--gamma', default=0.1, type=float, help='clustering loss weight')
parser.add_argument('--update_interval', default=100, type=int, help='update interval for target distribution')
parser.add_argument('--tol', default=0, type=float, help='stop criterium tolerance')
parser.add_argument('--crop', default=128, type=int, help='crop size of image')
parser.add_argument('--img_size', default=[3, 128, 128], nargs=3, type=int, help='size of custom images')
parser.add_argument('--leaky', default=True, type=str2bool)
parser.add_argument('--neg_slope', default=0.01, type=float)
parser.add_argument('--activations', default=False, type=str2bool)
parser.add_argument('--bias', default=True, type=str2bool)
parser.add_argument('--tensorboard', default=True, type=bool, help='export training stats to tensorboard')
# VAE parameters
parser.add_argument('--beta', default=1, type=float, help='Beta value for VAE')
args = parser.parse_args()
print(args)


T_vals = [246, 140, 100, 80, 60, 40]
for T in T_vals:
    args.update_interval = T
    if args.mode == 'pretrain' and not args.pretrain:
        print("Nothing to do :(")
        exit()

    # Deal with pretraining option and way of showing network path
    params = {'pretrain': args.pretrain}

    # Create directories structure
    dirs = ['runs', 'reports', 'nets']
    list(map(lambda x: os.makedirs(x, exist_ok=True), dirs))

    # Net architecture (model name) 
    model_name = args.dataset + '_' + args.net_architecture

    # Indexing (for automated reports saving) - allows to run many trainings and get all the reports collected.
    # If model name is already taken, a new idx number is added to new model for saving and logging. New model name will also be given if pretrained model is loaded. 
    reports_list = sorted(os.listdir('reports'), reverse=True)
    if reports_list:
        for file in reports_list:
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
    if not args.pretrain:
        pretrained = args.pretrained_net_path
    else:
        pretrained = os.path.join('nets', pretrained)
    if not args.pretrain and not os.path.isfile(pretrained):
        print("No pretrained weights, try again choosing pretrained network or create new with pretrain=True")

    model_files = [name_net, pretrained]
    params['model_files'] = model_files
    params['pic_path'] = name_pic
    # Open file
    if args.pretrain:
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
    if args.tensorboard:
        writer = SummaryWriter('runs/' + name)
        params['writer'] = writer
    else:
        params['writer'] = None


    # Hyperparameters
    params['batch'] = args.batch_size
    params['pretrain_epochs'] = args.epochs_pretrain
    params['print_freq'] = args.printing_frequency
    params['gamma'] = args.gamma
    params['update_interval'] = args.update_interval
    params['tol'] = args.tol

    # Loading data from datagenerator
    train_batch_size = val_batch_size = args.batch_size
    if args.dataset == 'MNIST' or args.dataset == 'CIFAR':
        dat_len = []
        data_transform = Compose([ ToTensor()])
        pre_loader, train_loader, val_loader = datagen.get_data_loaders(args.dataset, train_batch_size, val_batch_size, transform=data_transform)
        for data in pre_loader:
            _, labels = data
            dat_len.append(labels) 
        sampler = 3
        # train_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=True),
        #                             batch_size=args.batch_size, shuffle=False, sampler=sampler)

    if args.dataset == 'TNBC':
        train_transform = val_transform = Compose([
                                            transforms.CenterCrop(args.crop), 
                                            ToTensor()
                                            ])
        trainer = datagen.DataGen_TNBC(args.train_path, transform = train_transform)
        pre_loader = data.DataLoader(trainer, batch_size=train_batch_size, shuffle=True)
        # train_loader = data.DataLoader(trainer, batch_size=train_batch_size, shuffle=False)
        sampler = datagen.ManualRandomSampler(trainer)
        train_loader = data.DataLoader(trainer, sampler=sampler, batch_size=args.batch_size)
        val = datagen.DataGen_TNBC(args.eval_path, transform = val_transform)
        val_loader = data.DataLoader(val, batch_size=val_batch_size, shuffle=False)

    if args.dataset == 'Tumor':
        train_transform = Compose(([
                                    transforms.RandomHorizontalFlip(),
                                    ToTensor()
                                    ]))
        pre_trainer = datagen.DataGen_Tumor(args.train_path, 40000, transform=train_transform)
        trainer = datagen.DataGen_Tumor(args.train_path, 400, transform=train_transform)
        pre_loader = data.DataLoader(pre_trainer, batch_size=args.batch_size, shuffle=True)
        sampler = datagen.ManualRandomSampler(trainer.path)
        train_loader = data.DataLoader(trainer, sampler=sampler, batch_size=args.batch_size)

    if args.dataset == ('Tissues' or 'MNIST_subset'):
        train_transform = Compose([transforms.CenterCrop(args.img_size[2]),
                                    transforms.RandomHorizontalFlip(),
                                    ToTensor()])
        trainer = datagen.DataGen_Tissues(args.train_path, train_transform)
        pre_loader = data.DataLoader(trainer, batch_size=args.batch_size, shuffle=True)
        # train_loader = data.DataLoader(trainer, batch_size=args.batch_size, shuffle=False)
        sampler = datagen.ManualRandomSampler(trainer)
        train_loader = data.DataLoader(trainer, sampler=sampler, batch_size=args.batch_size)

    params['dataset_size'] = len(train_loader.dataset)
    params['img_size'] = args.img_size
    if args.crop <= args.img_size[1]:
        params['img_size'] = [3, args.crop, args.crop]
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
    tmp = "Batch size:\t" + str(args.batch_size)
    utils.print_both(f, tmp)
    tmp = "Clustering learning rate:\t" + str(args.rate)
    utils.print_both(f, tmp)
    tmp = "Pretraining learning rate:\t" + str(args.rate_pretrain)
    utils.print_both(f, tmp)
    tmp = "Weight decay:\t" + str(args.weight)
    utils.print_both(f, tmp)
    tmp = "Pretraining weight decay:\t" + str(args.weight_pretrain)
    utils.print_both(f, tmp)
    tmp = "Scheduler steps:\t" + str(args.sched_step)
    utils.print_both(f, tmp)
    tmp = "Scheduler gamma:\t" + str(args.sched_gamma)
    utils.print_both(f, tmp)
    tmp = "Pretraining scheduler steps:\t" + str(args.sched_step_pretrain)
    utils.print_both(f, tmp)
    tmp = "Pretraining scheduler gamma:\t" + str(args.sched_gamma_pretrain)
    utils.print_both(f, tmp)
    tmp = "Number of epochs of training:\t" + str(args.epochs)
    utils.print_both(f, tmp)
    tmp = "Number of epochs of pretraining:\t" + str(args.epochs_pretrain)
    utils.print_both(f, tmp)
    tmp = "Clustering loss weight:\t" + str(args.gamma)
    utils.print_both(f, tmp)
    tmp = "Update interval for target distribution:\t" + str(args.update_interval)
    utils.print_both(f, tmp)
    tmp = "Stop criterium tolerance:\t" + str(args.tol)
    utils.print_both(f, tmp)
    tmp = "Number of clusters:\t" + str(args.num_clusters)
    utils.print_both(f, tmp)
    tmp = "Leaky relu:\t" + str(args.leaky)
    utils.print_both(f, tmp)
    tmp = "Leaky slope:\t" + str(args.neg_slope)
    utils.print_both(f, tmp)
    tmp = "Activations:\t" + str(args.activations)
    utils.print_both(f, tmp)
    tmp = "Dataset:\t" + str(args.dataset)
    utils.print_both(f, tmp)
    tmp = "Image size:\t" + str(args.img_size)
    utils.print_both(f, tmp)
    tmp = "Bias:\t" + str(args.bias)
    utils.print_both(f, tmp)
    tmp = "z_dim:\t" + str(args.z_dim)
    utils.print_both(f, tmp)

    if args.net_architecture == 'TNBC_CAE_3' or 'TNBC_CAE_4' or 'TNBC_CAE_bn4' or 'TNBC_CAE_5' or 'TNBC_CAE_bn5':
        to_eval = "mods." + args.net_architecture + "(args.img_size, z_dim=args.z_dim, num_clusters=args.num_clusters)"
    elif args.net_architecture == 'MnistCAE':
        to_eval = "mods." + args.net_architecture + "()"
    elif args.net_architecture == 'VAE':
        to_eval = "mods." + args.net_architecture  + "(image_size=args.img_size[0]*args.img_size[1])"
    elif args.net_architecture == 'resnet18':
        to_eval = "mods." + args.net_architecture + "(args.img_size, args.z_dim)"
    elif args.net_architecture == 'VAE2':
        to_eval = "mods." + args.net_architecture + "(args)"
    elif args.net_architecture == 'VaDE':
        to_eval = 'mods.' + args.net_architecture + "(input_dim=args.img_size[0]*args.img_size[1]*args.img_size[2])"
    else:
        to_eval = "mods." + args.net_architecture + "(args.img_size, num_clusters=args.num_clusters, z_dim=args.z_dim, leaky = args.leaky, neg_slope = args.neg_slope)"

    model = eval(to_eval)
    model = model.to(device)

    # Optimizers and schedulers.
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.rate, weight_decay=args.weight)
    optimizer = optim.Adam(model.parameters(), lr=args.rate, weight_decay=args.weight)
    optimizer_pretrain = optim.Adam(model.parameters(), lr=args.rate_pretrain, weight_decay=args.weight_pretrain)
    optimizers = [optimizer, optimizer_pretrain]

    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.sched_step, gamma=args.sched_gamma)
    scheduler_pretrain = lr_scheduler.StepLR(optimizer_pretrain, step_size=args.sched_step_pretrain, gamma=args.sched_gamma_pretrain)
    schedulers = [scheduler, scheduler_pretrain]

    # Evaluate the proper model and optimizers given model_name
    if args.net_architecture == 'MnistCAE':
        criteria = nn.MSELoss(size_average=True)
        model = trainers.trainCAE(model, pre_loader, val_loader, criteria, optimizers[1], schedulers[1], args.pretrain_epochs, params)
    if args.net_architecture == 'resnet18':
        criteria = nn.MSELoss()
        model = trainers.resnet18(model, train_loader, criteria, optimizers[1], schedulers[1], args.pretrain_epochs, params)
    if args.net_architecture == 'VAE':
        model = trainers.trainVAE(model, train_loader, val_loader, optimizers[1], args.pretrain_epochs, params)
    if args.net_architecture == 'VAE2':
        model = trainers.trainVAE2(model, train_loader, val_loader, optimizers[1], params, args)
    if args.net_architecture == 'VaDE':
        model = trainers.trainVaDE(model, train_loader, val_loader, optimizers[1], args.pretrain_epochs, params, visualize=True)
    if args.net_architecture == 'CAE_3' or 'TNBC_CAE_bn3' or model_name == 'TNBC_CAE_3' or model_name == 'TNBC_CAE_4' or model_name == 'TNBC_CAE_bn4' or model_name == 'TNBC_CAE_5' or model_name == 'TNBC_CAE_bn5':
        criterion_1 = nn.MSELoss(size_average=True)     # Reconstruction loss
        criterion_2 = nn.KLDivLoss(reduction= 'batchmean')  # Clustering loss
        criteria = [criterion_1, criterion_2]
        if args.mode == 'train_full':
            model = trainers.train_model(model, pre_loader, train_loader, sampler, criteria, optimizers, schedulers, args.epochs, params)
            torch.save(model.state_dict(), name_net + '.pt')
        elif args.mode == 'pretrain':
            model = trainers.pretraining(model, pre_loader, criteria[0], optimizers[1], schedulers[1], args.epochs_pretrain, params)
