# trainers script for TNBC, Thomas Leicht Jensen Sep. 2019

import time
import os
import torch
import numpy as np
import copy
from sklearn.cluster import KMeans
from tqdm.autonotebook import tqdm
from torchvision.utils import save_image
from torch.autograd import Variable

import utils

# Training function (from my torch_DCEC implementation, kept for completeness)
def train_model(model, dataloader, criteria, optimizers, schedulers, num_epochs, params):

    # Note the time
    since = time.time()

    # Unpack parameters
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

    dl = dataloader

    # Pretrain or load weights
    if pretrain:
        while True:
            pretrained_model = pretraining(model, copy.deepcopy(dl), criteria[0], optimizers[1], schedulers[1], pretrain_epochs, params)
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
            utils.print_both(txt_file, 'Pretrained weights loaded from file: ' + str(pretrained))
        except:
            print("Couldn't load pretrained weights")

    # Initialise clusters
    utils.print_both(txt_file, '\nInitializing cluster centers based on K-means')
    kmeans(model, copy.deepcopy(dl), params)

    utils.print_both(txt_file, '\nBegin clusters training')

    # Prep variables for weights and accuracy of the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    # Initial target distribution
    utils.print_both(txt_file, '\nUpdating target distribution')
    output_distribution, labels, preds_prev = calculate_predictions(model, copy.deepcopy(dl), params)
    target_distribution = target(output_distribution)
    nmi = utils.metrics.nmi(labels, preds_prev)
    ari = utils.metrics.ari(labels, preds_prev)
    acc = utils.metrics.acc(labels, preds_prev)
    utils.print_both(txt_file,
                     'NMI: {0:.5f}\tARI: {1:.5f}\tAcc {2:.5f}\n'.format(nmi, ari, acc))

    if board:
        niter = 0
        writer.add_scalar('/NMI', nmi, niter)
        writer.add_scalar('/ARI', ari, niter)
        writer.add_scalar('/Acc', acc, niter)

    update_iter = 1
    finished = False

    # Go through all epochs
    for epoch in range(num_epochs):

        utils.print_both(txt_file, 'Epoch {}/{}'.format(epoch + 1, num_epochs))
        utils.print_both(txt_file,  '-' * 10)

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
            inputs, _ = data

            inputs = inputs.to(device)

            # Uptade target distribution, chack and print performance
            if (batch_num - 1) % update_interval == 0 and not (batch_num == 1 and epoch == 0):
                utils.print_both(txt_file, '\nUpdating target distribution:')
                output_distribution, labels, preds = calculate_predictions(model, dataloader, params)
                target_distribution = target(output_distribution)
                nmi = utils.metrics.nmi(labels, preds)
                ari = utils.metrics.ari(labels, preds)
                acc = utils.metrics.acc(labels, preds)
                utils.print_both(txt_file,
                                 'NMI: {0:.5f}\tARI: {1:.5f}\tAcc {2:.5f}\t'.format(nmi, ari, acc))
                if board:
                    niter = update_iter
                    writer.add_scalar('/NMI', nmi, niter)
                    writer.add_scalar('/ARI', ari, niter)
                    writer.add_scalar('/Acc', acc, niter)
                    update_iter += 1

                # check stop criterion
                delta_label = np.sum(preds != preds_prev).astype(np.float32) / preds.shape[0]
                preds_prev = np.copy(preds)
                if delta_label < tol:
                    utils.print_both(txt_file, 'Label divergence ' + str(delta_label) + '< tol ' + str(tol))
                    utils.print_both(txt_file, 'Reached tolerance threshold. Stopping training.')
                    finished = True
                    break

            tar_dist = target_distribution[((batch_num - 1) * batch):(batch_num*batch), :]
            tar_dist = torch.from_numpy(tar_dist).to(device)
            # print(tar_dist)

            # zero the parameter gradients
            optimizers[0].zero_grad()

            # Calculate losses and backpropagate
            with torch.set_grad_enabled(True):
                outputs, clusters, _ = model(inputs)
                loss_rec = criteria[0](outputs, inputs)
                loss_clust = gamma *criteria[1](torch.log(clusters), tar_dist) / batch
                loss = loss_rec + loss_clust
                loss.backward()
                optimizers[0].step()

            # For keeping statistics
            running_loss += loss.item() * inputs.size(0)
            running_loss_rec += loss_rec.item() * inputs.size(0)
            running_loss_clust += loss_rec.item() * inputs.size(0)

            # Some current stats
            loss_batch = loss.item()
            loss_batch_rec = loss_rec.item()
            loss_batch_clust = loss_clust.item()
            loss_accum = running_loss / ((batch_num - 1) * batch + inputs.size(0))
            loss_accum_rec = running_loss_rec / ((batch_num - 1) * batch + inputs.size(0))
            loss_accum_clust = running_loss_clust / ((batch_num - 1) * batch + inputs.size(0))

            if batch_num % print_freq == 0:
                utils.print_both(txt_file, 'Epoch: [{0}][{1}/{2}]\t'
                                           'Loss {3:.4f} ({4:.4f})\t'
                                           'Loss_recovery {5:.4f} ({6:.4f})\t'
                                           'Loss clustering {7:.4f} ({8:.4f})\t'.format(epoch + 1, batch_num,
                                                                                        len(dataloader),
                                                                                        loss_batch,
                                                                                        loss_accum, loss_batch_rec,
                                                                                        loss_accum_rec,
                                                                                        loss_batch_clust,
                                                                                        loss_accum_clust))
                if board:
                    niter = epoch * len(dataloader) + batch_num
                    writer.add_scalar('/Loss', loss_accum, niter)
                    writer.add_scalar('/Loss_recovery', loss_accum_rec, niter)
                    writer.add_scalar('/Loss_clustering', loss_accum_clust, niter)
            batch_num = batch_num + 1

            # Print image to tensorboard
            if batch_num == len(dataloader) and (epoch+1) % 5:
                inp = utils.tensor2img(inputs)
                out = utils.tensor2img(outputs)
                if board:
                    img = np.concatenate((inp, out), axis=1)
                    writer.add_image('Clustering/Epoch_' + str(epoch + 1).zfill(3) + '/Sample_' + str(img_counter).zfill(2), img)
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
                utils.print_both(txt_file, 'Pretraining:\tEpoch: [{0}][{1}/{2}]\t'
                           'Loss {3:.4f} ({4:.4f})\t'.format(epoch + 1, batch_num, len(dataloader),
                                                             loss_batch,
                                                             loss_accum))
                if board:
                    niter = epoch * len(dataloader) + batch_num
                    writer.add_scalar('Pretraining/Loss', loss_accum, niter)
            batch_num = batch_num + 1

            if batch_num in [len(dataloader), len(dataloader)//2, len(dataloader)//4, 3*len(dataloader)//4]:
                inp = utils.tensor2img(inputs)
                out = utils.tensor2img(outputs)
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
def calculate_predictions(model, dataloader, params):
    output_array = None
    label_array = None
    model.eval()
    for data in dataloader:
        inputs, labels = data
        inputs = inputs.to(params['device'])
        labels = labels.to(params['device'])
        _, outputs, _ = model(inputs)
        if output_array is not None:
            output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
            label_array = np.concatenate((label_array, labels.cpu().detach().numpy()), 0)
        else:
            output_array = outputs.cpu().detach().numpy()
            label_array = labels.cpu().detach().numpy()

    preds = np.argmax(output_array.data, axis=1)
    # print(output_array.shape)
    return output_array, label_array, preds

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
            
        
        # ----------------- VALIDATION  ----------------- 
        val_losses = 0
                
        print(f"Epoch {epoch+1}/{num_epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")
        # print_scores(precision, recall, f1, accuracy, val_batches)
        losses.append(total_loss/batches) # for plotting learning curve
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
                # z1 = z1.append(z)
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
                Y = []
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
                    Y.append(labels.numpy())
                    Y_pred.append(np.argmax(gamma, axis=1))
                    z1.append(z.detach().cpu().numpy())
                    if visualize and batch_idx == 0 and epoch % 10 == 0:
                        n = min(inputs.size(0), 8)
                        comparison = torch.cat([inputs.view(-1, 1, 28, 28)[:n],
                                                outputs.view(-1, 1, 28, 28)[:n]])
                        save_image(comparison.data.cpu(), os.path.join(img_path, 'reconstruction_{}.png'.format(epoch)), nrow=n)
                Y = np.concatenate(Y)
                Y_pred = np.concatenate(Y_pred)
                z1 = np.concatenate(z1)
                # view reconstruct
                    

                    
            
                # view sample
                if visualize:
                    sample = Variable(torch.randn(64, z_dim)).to(device)
                    sample = model.decode(sample)
                    save_image(sample.data.view(64, 1, 28, 28), os.path.join(img_path, 'sample_{}.png'.format(epoch)))
                    # variables = {'labels': Y, 'Y_pred': Y_pred, 'z': z1}
                    # np.save(os.path.join(img_path, 'VaDE_e{}'.format(epoch)), variables)
                np.save(os.path.join(img_path, 'labels_e{}'.format(epoch)), Y)
                np.save(os.path.join(img_path, 'preds_e{}'.format(epoch)), Y_pred)
                np.save(os.path.join(img_path, 'z_e{}'.format(epoch)), z1)
    return model