# trainers script for TNBC, Thomas Leicht Jensen Sep. 2019

import time
import os
import torch
import json
import numpy as np
import copy
import numbers
from torch import nn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as SHS
from sklearn.metrics import davies_bouldin_score as DBI 
from sklearn.metrics import calinski_harabasz_score as CHI
from tqdm.autonotebook import tqdm
from torchvision.utils import save_image
from torch.utils import data
from torch.autograd import Variable

import utils
import datagen

# Training function (from my torch_DCEC implementation, kept for completeness)
def train_model(model, pre_loader, dataloader, sampler, criteria, optimizers, schedulers, num_epochs, params):

    # Note the time
    since = time.time()
    writer = params['writer']
    if writer is not None: board = True
    txt_file = params['txt_file']
    pretrained = params['model_files'][1]
    pretrain = params['pretrain']
    print_freq = params['print_freq']
    dataset_size = params['dataset_size']
    device = params['device']
    batch = params['batch']
    pretrain_epochs = params['pretrain_epochs']
    gamma = params['gamma']
    update_interval = params['update_interval']
    tol = params['tol']
    img_size = params['img_size']
    
    dl = dataloader

    # Pretrain or load weights
    if pretrain:
        while True:
            pretrained_model = pretraining(model, pre_loader, criteria[0], optimizers[1], schedulers[1], pretrain_epochs, params)
            if pretrained_model:
                break
            else:
                for layer in model.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
        model = pretrained_model
    else:
        try:
            model.load_state_dict(torch.load(pretrained))
            utils.print_both(txt_file, 'Pretrained weights loaded from model: ' + str(pretrained))
        except:
            print("Couldn't load pretrained weights")

    # Initialise clusters
    utils.print_both(txt_file, '\nInitializing cluster centers based on K-means')
    kmeans(model, copy.deepcopy(dl), params)

    utils.print_both(txt_file, '\nBegin clusters training')

    # Prep variables for weights and accuracy of the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    # Note Thomas: If going back to shuffle=False then uncomment lin 70-84
    # Initial target distribution
    # utils.print_both(txt_file, '\nUpdating target distribution')
    # _, _, output_distribution, z, labels, preds_prev = calculate_predictions(model, dataloader)
    # target_distribution = target(output_distribution)
    # pca_init = PCA(2, random_state=71991).fit_transform(z)
    # x_lim = (np.min(pca_init[:,0]), np.max(pca_init[:,0])) 
    # y_lim = ((np.min(pca_init[:,1]), np.max(pca_init[:,1])))
    # utils.PCA_plot(pca_init, labels, preds_prev, x_lim, y_lim, params['pic_path'], 'distr_epoch_0')
    # KL_div = round(criteria[1](torch.log(torch.from_numpy(output_distribution).to(device)), torch.from_numpy(target_distribution).to(device)).item(), 4)
    # utils.plot_distributions(output_distribution, target_distribution, KL_div, os.path.join(params['pic_path'], 'init_q_p'))
    # if isinstance(labels, np.ndarray):
    #     nmi = utils.metrics.nmi(labels, preds_prev, average_method='arithmetic')
    #     ari = utils.metrics.ari(labels, preds_prev)
    #     acc = utils.metrics.acc(labels, preds_prev)
    #     utils.print_both(txt_file,
    #                     'NMI: {0:.5f}\tARI: {1:.5f}\tAcc {2:.5f}\n'.format(nmi, ari, acc))

    #     if board:
    #         niter = 0
    #         writer.add_scalar('/NMI', nmi, niter)
    #         writer.add_scalar('/ARI', ari, niter)
    #         writer.add_scalar('/Acc', acc, niter)
            
    update_iter = 1
    finished = False

    loss_clust = prev_clust =  torch.zeros((1))
    outputs = inputs = torch.rand((batch, img_size[0], img_size[1], img_size[2]))
    # Go through all epochs
    for epoch in range(num_epochs):
        # Note Thomas: remove lin. 101-108 if shuffle=False
        utils.print_both(txt_file, 'Epoch {}/{}'.format(epoch + 1, num_epochs))
        utils.print_both(txt_file,  '-' * 10)
        if epoch > 0:
            sampler.shuffle()
        utils.print_both(txt_file, '\nUpdating target distribution for new epoch:')
        _, _, output_distribution, z, labels, preds_prev = calculate_predictions(model, dl)
        preds = preds_prev
        target_distribution = target(output_distribution)
        if epoch==0:
            pca_init = PCA(2, random_state=71991).fit_transform(z)
            x_lim = (np.min(pca_init[:,0]), np.max(pca_init[:,0])) 
            y_lim = ((np.min(pca_init[:,1]), np.max(pca_init[:,1])))
        # if isinstance(labels, np.ndarray):
        #     nmi = utils.metrics.nmi(labels, preds_prev, average_method='arithmetic')
        #     ari = utils.metrics.ari(labels, preds_prev)
        #     acc = utils.metrics.acc(labels, preds_prev)
        #     writer.add_scalar('/NMI', nmi, niter)
        #     writer.add_scalar('/ARI', ari, niter)
        #     writer.add_scalar('/Acc', acc, niter)

        schedulers[0].step()
        model.train(True)  # Set model to training mode

        running_loss = 0.0
        running_loss_rec = 0.0
        running_loss_clust = 0.0
        # Keep the batch number for inter-phase statistics
        batch_num = 1
        img_counter = 0

        # Iterate over data.
        for data in dataloader:
            # Get the inputs and labels
            prev_inputs = inputs
            prev_outputs = outputs
            inputs, _ = data
            inputs = inputs.to(device)
            # Update target distribution, check and print performance
            if (((batch_num - 1) % update_interval == 0) and not batch_num==1) and not (batch_num == 1 and epoch == 0): # Note, Thomas: Remove "and not batch_num==1" in first statement to go back to shuffle False dataloader
                utils.print_both(txt_file, '\nUpdating target distribution:')
                _, _, output_distribution, z, _, preds = calculate_predictions(model, dataloader)
                target_distribution = target(output_distribution)
                niter = update_iter
                if isinstance(labels, np.ndarray):
                    nmi = utils.metrics.nmi(labels, preds, average_method='arithmetic')
                    ari = utils.metrics.ari(labels, preds)
                    acc = utils.metrics.acc(labels, preds)
                    utils.print_both(txt_file,
                                    'NMI: {0:.5f}\tARI: {1:.5f}\tAcc {2:.5f}\t'.format(nmi, ari, acc))
                    writer.add_scalar('/NMI', nmi, niter)
                    writer.add_scalar('/ARI', ari, niter)
                    writer.add_scalar('/Acc', acc, niter)
                    KL_div = round(criteria[1](torch.log(torch.from_numpy(output_distribution).to(device)), torch.from_numpy(target_distribution).to(device)).item(), 4)
                    # utils.plot_distributions(output_distribution, target_distribution, KL_div, os.path.join(params['pic_path'], 'pq_epoch_' + str(epoch+1).zfill(3) + 'T_' + str(niter).zfill(3)))
                # logging weights, gradients, nan and infs during training
                # for tag, value in model.named_parameters():
                #     tag = tag.replace('.', '/')
                #     if np.isnan(value.data.cpu().numpy()).any() or np.isnan(value.grad.data.cpu().numpy()).any():
                #         print('NaN detected at epoch: '+str(epoch+1) + ', T=' + str(niter) + ' in layer: ' + tag[:-7])
                #         writer.add_histogram(tag+'/NaN', value.data.cpu().numpy(), niter)
                #         writer.add_histogram(tag+'/grad/NaN', value.grad.data.cpu().numpy(), niter)
                #     if np.isinf(value.data.cpu().numpy()).any() or np.isinf(value.grad.data.cpu().numpy()).any():
                #         print('Inf detected at epoch: '+str(epoch+1) + ', T=' + str(niter) + ' in layer: ' + tag[:-7])
                #         writer.add_histogram(tag+'/Inf', value.data.cpu().numpy(), niter)
                #         writer.add_histogram(tag+'/grad/Inf', value.grad.data.cpu().numpy(), niter)
                #     writer.add_histogram(tag, value.data.cpu().numpy(), niter)
                #     writer.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), niter)

                update_iter += 1

                # check stop criterion
                if len(np.unique(preds))<2:
                    utils.print_both(txt_file, 'Pseudo convergence: Only one class predicted. Stopping training.')
                    finished=True
                    break
                chi_score = CHI(z, preds)
                silhouette_score = SHS(z, preds)
                dbi_score = DBI(z, preds)
                writer.add_scalar('Internals/CHI', chi_score, niter)
                writer.add_scalar('Internals/Silhouette', silhouette_score, niter)
                writer.add_scalar('Internals/DBI', dbi_score, niter)
                delta_label = np.sum(preds != preds_prev).astype(np.float32) / preds.shape[0]
                preds_prev = np.copy(preds)
                writer.add_scalar('Clustering/delta', delta_label, niter)
                if delta_label < tol:
                    utils.print_both(txt_file, 'Label divergence ' + str(delta_label) + '< tol ' + str(tol))
                    pca = PCA(2, random_state=71991).fit_transform(z)
                    
                    if isinstance(labels, np.ndarray):
                        utils.PCA_plot(pca, labels, preds, x_lim, y_lim, params['pic_path'], 'final_distr_epoch_'+str(epoch+1) + 'T_' + str(niter))
                    else:
                        utils.TNBC_PCA(pca, preds, x_lim, y_lim, params['pic_path'], 'final_distr_epoch_'+str(epoch+1))
                    utils.print_both(txt_file, 'Reached tolerance threshold. Stopping training.')
                    finished = True
                    break

            tar_dist = target_distribution[((batch_num - 1) * batch):(batch_num*batch), :]
            tar_dist = torch.from_numpy(tar_dist).to(device)
            
            # zero the parameter gradients
            optimizers[0].zero_grad()

            # Calculate losses and backpropagate
            with torch.set_grad_enabled(True):
                outputs, clusters, _ = model(inputs)
                loss_rec = criteria[0](outputs, inputs)
                loss_clust = criteria[1](torch.log(clusters), tar_dist)
                loss = loss_rec + gamma*loss_clust
                loss.backward() 
                optimizers[0].step()

            # For keeping statistics
            running_loss += loss.item() * inputs.size(0)
            running_loss_rec += loss_rec.item() * inputs.size(0)
            running_loss_clust += gamma*loss_clust.item() * inputs.size(0)

            # Some current stats
            loss_batch = loss.item()
            loss_batch_rec = loss_rec.item()
            loss_batch_clust = loss_clust.item()
            loss_accum = running_loss / ((batch_num - 1) * batch + inputs.size(0))
            loss_accum_rec = running_loss_rec / ((batch_num - 1) * batch + inputs.size(0))
            loss_accum_clust = running_loss_clust / ((batch_num - 1) * batch + inputs.size(0))

            if batch_num % print_freq == 0:
                niter = epoch * len(dataloader) + batch_num
                writer.add_scalar('/Loss', loss_accum, niter)
                writer.add_scalar('/Loss_recovery', loss_accum_rec, niter)
                writer.add_scalar('/Loss_clustering', loss_accum_clust, niter)
                writer.add_scalar('Clustering/L_c', loss_clust.item(), niter)
                writer.add_scalar('Clustering/L_r', loss_rec.item(), niter)
                writer.add_scalar('Clustering/L_tot', loss.item(), niter)
                d_Lc = loss_clust.item()-prev_clust.item()
                writer.add_scalar('Clustering/delta_L_c', d_Lc, niter)
                prev_clust = loss_clust
                # # Sample n images from all batches for a few epochs only (otherwise takes too much storage)
                # n = min(inputs.size(0), 10)
                # perm = torch.randperm(inputs.size(0))
                # idx = perm[:n]
                # batch_img = torch.cat([inputs.view(-1, 3, 128, 128)[idx],
                #                         outputs.view(-1, 3, 128, 128)[idx]])
                # save_image(batch_img.data.cpu(),os.path.join(params['pic_path'], 'reconstruction_epoch_' + str(epoch+1) + '_batch_' + str(niter) + '.png'), nrow=n)
                if abs(d_Lc) > 0.2:
                    path = os.path.join(params['pic_path'], 'Spiked')
                    os.makedirs(path, exist_ok=True)
                    n = min(inputs.size(0), 5)
                    comparison = torch.cat([inputs.view(-1, 3, 128, 128)[:n],
                                        outputs.view(-1, 3, 128, 128)[:n]])
                    prev_comparison = torch.cat([prev_inputs.view(-1, 3, 128, 128)[:n], 
                                        prev_outputs.view(-1, 3, 128, 128)[:n]])
                    save_image(comparison.data.cpu(),os.path.join(path, 'reconstruction_epoch_' + str(epoch+1) + '_batch_' + str(niter) + '.png'), nrow=n)
                    save_image(prev_comparison.data.cpu(), os.path.join(path, 'reconstruction_epoch' + str(epoch+1) + '_batch_' + str(niter-1) + '.png'), nrow=n)
                    pca = PCA(2, random_state=71991).fit_transform(z)
                    if isinstance(labels, np.ndarray):
                        utils.PCA_plot(pca, labels, preds_prev, x_lim, y_lim, path, 'spiked_PCA_e'+str(epoch+1) + 'batch_' + str(niter) + 'T_'+ str(update_iter))
                    else:
                        utils.TNBC_PCA(pca, preds_prev, x_lim, y_lim, path, 'spiked_PCA_e'+str(epoch+1) + 'batch_' + str(niter) + 'T_'+ str(update_iter))
                    KL_div = round(criteria[1](torch.log(torch.from_numpy(output_distribution).to(device)), torch.from_numpy(target_distribution).to(device)).item(), 4)
                    utils.plot_distributions(output_distribution, target_distribution, KL_div, os.path.join(path, 'q_p_epoch_' + str(epoch+1) + ' batch_' + str(niter) + 'T_' + str(update_iter)))

            batch_num = batch_num + 1

            # Print image to tensorboard
            if batch_num == len(dataloader):
                inp = utils.tensor2img(inputs)
                out = utils.tensor2img(outputs)
                n = min(inputs.size(0), 5)
                img2 = torch.cat([inputs.view(-1, 3, 128, 128)[:n],
                                        outputs.view(-1, 3, 128, 128)[:n]])
                img = np.concatenate((inp, out), axis=1)
                writer.add_image('Clustering/Epoch_' + str(epoch + 1).zfill(3) + '/Sample_' + str(img_counter).zfill(2), img)
                save_image(img2.data.cpu(),os.path.join(params['pic_path'], 'reconstruction_epoch_' + str(epoch+1) + ' batch_' + str(niter) + '.png'), nrow=n)
                pca = PCA(2, random_state=71991).fit_transform(z)
                if isinstance(labels, np.ndarray):
                    utils.PCA_plot(pca, labels, preds, x_lim, y_lim, params['pic_path'], 'distr_epoch_'+str(epoch+1))
                else:
                    utils.TNBC_PCA(pca, preds, x_lim, y_lim, params['pic_path'], 'distr_epoch_'+str(epoch+1))
                img_counter += 1
            

        if finished: break

        epoch_loss = running_loss / dataset_size
        epoch_loss_rec = running_loss_rec / dataset_size
        epoch_loss_clust = running_loss_clust / dataset_size

        if board:
            writer.add_scalar('/Loss' + '/Epoch', epoch_loss, epoch + 1)
            writer.add_scalar('/Loss_rec' + '/Epoch', epoch_loss_rec, epoch + 1)
            writer.add_scalar('/Loss_clust' + '/Epoch', epoch_loss_clust, epoch + 1)

        utils.print_both(txt_file, 'Loss: {0:.4f}\tLoss_recovery: {1:.4f}\tLoss_clustering: {2:.4f}'.format(epoch_loss,
                                                                                                            epoch_loss_rec,
                                                                                                            epoch_loss_clust))

        # If wanted to do some criterium in the future (for now useless)
        if epoch_loss < best_loss or epoch_loss >= best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        utils.print_both(txt_file, '')
        # here: shuffle dataloader

    time_elapsed = time.time() - since
    utils.print_both(txt_file, 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Pretraining function for recovery loss only
def pretraining(model, dataloader, criterion, optimizer, scheduler, num_epochs, params):
    # Note the time
    since = time.time()

    # Unpack parameters
    writer = params['writer']
    if writer is not None: board = True
    txt_file = params['txt_file']
    pretrained = params['model_files'][1]
    print_freq = params['print_freq']
    dataset_size = params['dataset_size']
    device = params['device']
    batch = params['batch']
    img_size = params['img_size']
    # Prep variables for weights and accuracy of the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    # Go through all epochs
    for epoch in range(num_epochs):
        utils.print_both(txt_file, 'Pretraining:\tEpoch {}/{}'.format(epoch + 1, num_epochs))
        utils.print_both(txt_file, '-' * 10)

        scheduler.step()
        model.train(True)  # Set model to training mode

        running_loss = 0.0

        # Keep the batch number for inter-phase statistics
        batch_num = 1
        # Images to show
        img_counter = 0

        # Iterate over data.
        for data in dataloader:
            # Get the inputs and labels
            inputs, _ = data
            inputs = inputs.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs, _, _ = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()

            # For keeping statistics
            running_loss += loss.item() * inputs.size(0)

            # Some current stats
            loss_batch = loss.item()
            loss_accum = running_loss / ((batch_num - 1) * batch + inputs.size(0))

            if batch_num % print_freq == 0:
                # utils.print_both(txt_file, 'Pretraining:\tEpoch: [{0}][{1}/{2}]\t'
                #            'Loss {3:.4f} ({4:.4f})\t'.format(epoch + 1, batch_num, len(dataloader),
                #                                              loss_batch,
                #                                              loss_accum))
                if board:
                    niter = epoch * len(dataloader) + batch_num
                    writer.add_scalar('Pretraining/Loss', loss_accum, niter)
            batch_num = batch_num + 1

            if batch_num == len(dataloader):
                inp = utils.tensor2img(inputs)
                out = utils.tensor2img(outputs)
                n = min(inputs.size(0), 4)
                if img_size[2]==1:
                    comparison = torch.cat([inputs.view(-1, img_size[2], img_size[0], img_size[1])[:n],
                                        outputs.view(-1, img_size[2], img_size[0], img_size[1])[:n]])
                if img_size[0]==3:    
                    comparison = torch.cat([inputs.view(-1, 3, img_size[1], img_size[2])[:n],
                                            outputs.view(-1, 3, img_size[1], img_size[2])[:n]])
                save_image(comparison.data.cpu(),os.path.join(params['pic_path'], 'reconstruction_' + str(epoch) + '.png'), nrow=n)
                if board:
                    img = np.concatenate((inp, out), axis=1)
                    writer.add_image('Pretraining/Epoch_' + str(epoch + 1).zfill(3) + '/Sample_' + str(img_counter).zfill(2), img)
                    img_counter += 1

        epoch_loss = running_loss / dataset_size
        if epoch == 0: first_loss = epoch_loss
        if epoch == 4 and epoch_loss / first_loss > 1:
            utils.print_both(txt_file, "\nLoss not converging, starting pretraining again\n")
            return False

        if board:
            writer.add_scalar('Pretraining/Loss' + '/Epoch', epoch_loss, epoch + 1)

        utils.print_both(txt_file, 'Pretraining:\t Loss: {:.4f}'.format(epoch_loss))

        # If wanted to add some criterium in the future
        if epoch_loss < best_loss or epoch_loss >= best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        utils.print_both(txt_file, '')

    time_elapsed = time.time() - since
    utils.print_both(txt_file, 'Pretraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    model.pretrained = True
    torch.save(model.state_dict(), pretrained)

    return model

# K-means clusters initialisation
def kmeans(model, dataloader, params):
    km = KMeans(n_clusters=model.num_clusters, n_init=20)
    output_array = None
    model.eval()
    # Itarate throught the data and concatenate the latent space representations of images
    for data in dataloader:
        inputs, _ = data
        inputs = inputs.to(params['device'])
        _, _, outputs = model(inputs)
        if output_array is not None:
            output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
        else:
            output_array = outputs.cpu().detach().numpy()
        # print(output_array.shape)
        if output_array.shape[0] > 50000: break

    # Perform K-means
    km.fit_predict(output_array)
    # Update clustering layer weights
    weights = torch.from_numpy(km.cluster_centers_)
    model.clustering.set_weight(weights.to(params['device']))
    # torch.cuda.empty_cache()

# Function forwarding data through network, collecting clustering weight output and returning prediciotns and labels
def calculate_predictions(model, dataloader):
    output_array = None
    label_array = None
    z = []
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for data in dataloader:
        inputs, labels = data[0].to(device), data[1]
        if isinstance(labels, torch.Tensor):
               labels = labels.to(device)
        x_tilde, outputs, emb = model(inputs)
        z.append(emb.detach().cpu().numpy())
        if output_array is not None:
            output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
            if isinstance(label_array, np.ndarray):
                label_array = np.concatenate((label_array, labels.cpu().detach().numpy()), 0)
            else: 
                label_array = label_array + labels
        else:
            output_array = outputs.cpu().detach().numpy()
            if isinstance(labels, torch.Tensor):
                label_array = labels.cpu().detach().numpy()
            else:
                label_array = labels
    z = np.concatenate(z)
    preds = np.argmax(output_array.data, axis=1) 
    return x_tilde, inputs, output_array, z, label_array, preds

# Calculate target distribution
def target(out_distr):
    tar_dist = out_distr ** 2 / np.sum(out_distr, axis=0)
    tar_dist = np.transpose(np.transpose(tar_dist) / np.sum(tar_dist, axis=1))
    return tar_dist

def trainCAE(model, train_loader, val_loader, criterion, optimizer, schedulers, num_epochs, params):
    val_batches = params['batch']
    batches = params['batch']
    device = params['device']
    img_path = params['pic_path']
    writer = params['writer']
    if writer is not None: board = True
    img_path = os.path.join(os.getcwd(), img_path)
    losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        # progress bar (works in Jupyter notebook too!)
        progress = tqdm(enumerate(train_loader), desc="Loss: ", total=len(train_loader))

        model.train()
        
        for i, data in progress:
            X, y = data[0].to(device), data[1]
            if y is isinstance(y, torch.Tensor):
               y = y.to(device)
            # training step for single batch
            model.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, X)
            loss.backward()
            optimizer.step()

            # getting training quality data
            current_loss = loss.item()
            total_loss += current_loss

            # updating progress bar
            progress.set_description("Loss: {:.4f}".format(total_loss/(i+1)))
        if board:    
            writer.add_scalar('Pretraining/Loss', total_loss/batches, total_loss)
        # ----------------- VALIDATION  ----------------- 
        val_losses = 0
                
        print(f"Epoch {epoch+1}/{num_epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")
        if epoch % 10 == 0:
            save_image(outputs, os.path.join(img_path, 'AEImage_{}.png'.format(epoch)))
        if epoch == num_epochs-1:
            z1 = []
            labels = []
            for i, data in enumerate(val_loader):
                X, y = data[0].to(device), data[1]
                if y is isinstance(y, torch.Tensor):
                    y = y.to(device) 
                z = model.embed(X)
                labels.append(y)
                z = z.detach().cpu().numpy()
                z1.append(z)
            z1 = np.concatenate(z1)
            labels = np.concatenate(labels)
            np.save(os.path.join(img_path, 'latrep'), z1)
            np.save(os.path.join(img_path, 'labels'), labels)
    return model

def trainVAE(model, train_loader, val_loader, optimizer, num_epochs, params):
    img_path = params['pic_path']
    img_path = os.path.join(os.getcwd(), img_path)
    device = params['device']
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            img = data[0].to(device)
            img = img.view(img.size(0), -1)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(img)
            loss = model.loss_function(recon_batch, img, mu, logvar)
            loss.backward()
            train_loss += loss.data.detach().cpu().numpy()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(img),
                    len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    train_loss / len(img)))
        if epoch % 10 == 0:
            save_image(recon_batch, os.path.join(img_path, 'VAE_recon_{}.png'.format(epoch)))
            np.save(os.path.join(img_path, 'VAE_recon_{}'.format(epoch)), recon_batch.detach().cpu().numpy())
        if epoch == num_epochs-1:            
            fc2 = []
            fc3 = []
            mu_val = []
            var_val = []
            for i, data in enumerate(val_loader):
                X, _ = data[0].to(device), data[1].to(device)
                X = X.view(X.size(0), -1) 
                fc_2, fc_3 = model.encode(X)
                _, mu1, var1 = model(X)
                fc2.append(fc_2)
                fc3.append(fc_3)
                mu_val.append(mu1)
                var_val.append(var1)
            np.save(os.path.join(img_path, 'fc2'), fc2)
            np.save(os.path.join(img_path, 'fc3'), fc3)
            np.save(os.path.join(img_path, 'mu'), mu_val)
            np.save(os.path.join(img_path, 'var'), var_val)
    return model

def trainVAE2(model, train_loader, val_loader, optimizers, params, args):
    num_epochs = params['pretrain_epochs']
    device = params['device']
    img_path = params['pic_path']
    img_path = os.path.join(os.getcwd(), img_path)
    batches = params['batch']
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        total_loss = 0
        progress = tqdm(enumerate(train_loader), desc="Loss: ", total=len(train_loader))
        for batch_idx, data in progress:
            img, labels = data[0].to(device), data[1].to(device)
            optimizers.zero_grad()
            loss = model.loss_function(img)
            loss.backward()
            train_loss += loss.data.detach().cpu().numpy()

            optimizers.step()
            current_loss = loss.item()
            total_loss += current_loss
            progress.set_description("Loss: {:.4f}".format(current_loss), "batch_idx: {}".format(batch_idx))
        print(f"Epoch {epoch+1}/{num_epochs}, training loss: {total_loss}")
        # progress_val = tqdm(enumerate(val_loader), desc="Loss: ", total=batches)
        if epoch == num_epochs-1:            
            z1 = []
            labels = []
            for i, data in enumerate(val_loader):
                X, y = data[0].to(device), data[1].to(device)
                _, _, _, z = model(X)

                z1.append(z.detach().cpu().numpy())
                labels.append(y.detach().cpu().numpy())
            np.save(os.path.join(img_path, 'labels'), np.concatenate(labels))    
            np.save(os.path.join(img_path, 'z'), np.concatenate(z1))
    return model

def trainVaDE(model, train_loader, val_loader, optimizer, num_epochs, params, visualize=True, z_dim=20):

    device = params['device']
    img_path = params['pic_path']
    batches = params['batch']
    img_size = params['img_size']
    img_path = os.path.join(os.getcwd(), img_path)
    for epoch in range(num_epochs):
            progress = tqdm(enumerate(train_loader), desc="Loss: ", total=len(train_loader))
            train_loss = 0
            model.train()
            
            for batch_idx, (inputs, _) in progress:
                inputs = inputs.view(inputs.size(0), -1).float().to(device)
                # if use_cuda:
                #     inputs = inputs.cuda()
                optimizer.zero_grad()
                inputs = Variable(inputs)
                
                z, outputs, mu, logvar = model.forward(inputs)
                loss = model.loss_function(outputs, inputs, z, mu, logvar)
                # train_loss += loss.data[0]*len(inputs)
                train_loss += loss.data.detach().cpu().numpy()*len(inputs)
                loss.backward()
                optimizer.step()
                progress.set_description("Loss: {:.4f}".format(loss), "batch_idx: {}".format(batch_idx))
            print(f"Epoch {epoch+1}/{num_epochs}, training loss: {train_loss}")

            if epoch % 25 == 0 or epoch == num_epochs-1:
                # validate
                valid_loss = 0.0
                Y = None
                Y_pred = []
                z1 = []
                progress_val = tqdm(enumerate(val_loader), desc="Loss", total=len(val_loader))
                for batch_idx, (inputs, labels) in progress_val:
                    inputs = inputs.view(inputs.size(0), -1).float().to(device)
                    inputs = Variable(inputs)
                    z, outputs, mu, logvar = model.forward(inputs)

                    loss = model.loss_function(outputs, inputs, z, mu, logvar)
                    valid_loss += loss.data.detach().cpu().numpy()*len(inputs)
                    progress.set_description("Loss: {:.4f}".format(loss), "batch_idx: {}".format(batch_idx))
                    gamma = model.get_gamma(z, mu, logvar).data.cpu().numpy()
                    if Y is None:
                        if isinstance(labels, np.ndarray):
                            Y.append(labels.numpy())
                        else:
                            Y = labels
                    else:
                        if isinstance(labels, np.ndarray):
                            Y.append(labels.cpu().detach().numpy())
                        else:
                            Y = Y + labels
                    Y_pred.append(np.argmax(gamma, axis=1))
                    z1.append(z.detach().cpu().numpy())
                    if visualize and batch_idx == 0 and epoch % 10 == 0:
                        n = min(inputs.size(0), 8)
                        comparison = torch.cat([inputs.view(-1, img_size[0], img_size[1], img_size[2])[:n],
                                                outputs.view(-1, img_size[0], img_size[1], img_size[2])[:n]])
                        save_image(comparison.data.cpu(), os.path.join(img_path, 'reconstruction_{}.png'.format(epoch)), nrow=n)
                if isinstance(Y, np.ndarray):
                    Y = np.concatenate(Y)
                Y_pred = np.concatenate(Y_pred)
                z1 = np.concatenate(z1)
                # view reconstruct
            
                # view sample
                if visualize:
                    sample = Variable(torch.randn(64, z_dim)).to(device)
                    sample = model.decode(sample)
                    save_image(sample.data.view(64, img_size[0], img_size[1], img_size[2]), os.path.join(img_path, 'sample_{}.png'.format(epoch)))
                    # variables = {'labels': Y, 'Y_pred': Y_pred, 'z': z1}
                    # np.save(os.path.join(img_path, 'VaDE_e{}'.format(epoch)), variables)
                np.save(os.path.join(img_path, 'labels_e{}'.format(epoch)), Y)
                np.save(os.path.join(img_path, 'preds_e{}'.format(epoch)), Y_pred)
                np.save(os.path.join(img_path, 'z_e{}'.format(epoch)), z1)
    return model

def resnet18(model, train_loader, criterion, optimizer, scheduler, params):
    writer = params['writer']
    device = params['device']
    pretrain_epochs = params['pretrain_epochs']
    if writer is not None: board = True
    for epoch in range(pretrain_epochs):
        run_loss = 0
        img_counter = 0
        scheduler.step()
        progress = tqdm(enumerate(train_loader), desc="Loss: ", total=len(train_loader))

        model.train(True)
        
        for i, data in progress:
            X, y = data[0].to(device), data[1]
            if y is isinstance(y, torch.Tensor):
               y = y.to(device)
            optimizer.zero_grad()
            x_tilde, z = model(X)
            loss = criterion(x_tilde, X)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            if (i+1) == len(train_loader):
                print('Epoch: [%d/%d] Autoencoder loss: %.3f' % (epoch + 1, pretrain_epochs, run_loss/2))

                n = min(X.size(0), 8)
                comparison = torch.cat([X.view(-1, 3, 128, 128)[:n],
                                        x_tilde.view(-1, 3, 128, 128)[:n]])
                save_image(comparison.data.cpu(),os.path.join(params['pic_path'], 'reconstruction_' + str(epoch) + '.png'), nrow=n)

                if board:
                    img = np.concatenate((utils.tensor2img(X), utils.tensor2img(x_tilde)), axis=1)
                    writer.add_image('Pretraining/Epoch_' + str(epoch + 1).zfill(3) + '/Sample_' + str(img_counter).zfill(2), img)
                    writer.add_scalar('Pretraining/Loss_recon' + '/Epoch', run_loss, epoch + 1) 
                    img_counter += 1
    return model