# # Evaluation script, Thomas
if __name__ == "__main__":
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
    import argparse
    from torch import nn, optim
    import numpy as np
    import pandas as pd

    import mods
    import datagen
    import trainers
    import utils

    # NMI, ARI and ACC scores from DCEC with CAE_3 
    # train_batch_size = val_batch_size = 256
    # _, val_loader = datagen.get_data_loaders('MNIST-train', train_batch_size, val_batch_size)
    # model_name = 'CAE_3'
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # params = {'device': device}
    # weights_path = os.path.abspath(r"C:\Source\Research Repositories\TNBC\nets\CAE_3_001.pt")
    # img_size = [28, 28, 1]
    # n_clusters = 10
    # to_eval = "mods." + model_name + "(img_size, num_clusters=n_clusters)"
    
    # model = eval(to_eval)
    # model = model.to(device)
    # model.load_state_dict(torch.load(weights_path))    
    # output_array, label_array, preds = trainers.calculate_predictions(model, val_loader, params)
    # nmi_DCAE3 = utils.metrics.nmi(label_array, preds)
    # ari_DCAE3 = utils.metrics.ari(label_array, preds)
    # acc_DCAE3 = utils.metrics.acc(label_array, preds)

    # plotting PCA on DCEC CAE_3 latrep
    # utils.PCA_plot(output_array, label_array, preds, model_name, n_clusters) 
    # t-sne
    # utils.tsne_plot(output_array, label_array, preds, model_name, n_clusters)

    # CAE_3 pretrained (no clustering layer)
    # path_pretrained = os.path.abspath(r"C:\Source\Research Repositories\TNBC\nets\CAE_3_001_pretrained.pt")
    # pretrained_name = 'CAE_3'
    # pre_eval = "mods." + model_name + "(img_size, num_clusters=n_clusters)"
    # pre_model = eval(pre_eval)
    # pre_model = pre_model.to(device)
    # pre_model.load_state_dict(torch.load(path_pretrained))
    # pretrained_array, pre_label_array, pre_preds = trainers.calculate_predictions(pre_model, val_loader, params)
    # nmi_pre3 = utils.metrics.nmi(pre_label_array, pre_preds)
    # ari_pre3 = utils.metrics.ari(pre_label_array, pre_preds)
    # acc_pre3 = utils.metrics.acc(pre_label_array, pre_preds)
    
    # plotting PCA on DCEC CAE_3 latrep
    # utils.PCA_plot(pretrained_array, pre_label_array, pre_preds, pretrained_name, n_clusters) 
    # t-sne
    # utils.tsne_plot(pretrained_array, pre_label_array, pre_preds, pretrained_name, n_clusters)

    # CAE_bn3 
    # path_dcec_bn3 = os.path.abspath(r"C:\Source\Research Repositories\TNBC\nets\CAE_bn3_001.pt")
    # cae_bn3_name = 'CAE_bn3'
    # cae_bn3_eval = "mods." + cae_bn3_name + "(img_size, n_clusters)"
    # model_cae_bn3 = eval(cae_bn3_eval)
    # model_cae_bn3 = model_cae_bn3.to(device)
    # model_cae_bn3.load_state_dict(torch.load(path_dcec_bn3))

    # cae_bn3_array, cae_bn3_label_array, cae_bn3_preds = trainers.calculate_predictions(model_cae_bn3, val_loader, params)
    # nmi_cae_bn3 = utils.metrics.nmi(label_array, preds)
    # ari_cae_bn3 = utils.metrics.ari(label_array, preds)
    # acc_cae_bn3 = utils.metrics.acc(label_array, preds)
    
    # plotting PCA on DCEC CAE_3 latrep
    # utils.PCA_plot(cae_bn3_array, cae_bn3_label_array, cae_bn3_preds, cae_bn3_name, n_clusters) 
    # t-sne
    # utils.tsne_plot(cae_bn3_array, cae_bn3_label_array, cae_bn3_preds, cae_bn3_name, n_clusters)


    # CAE_bn3 pretrained (no clustering layer)
    # cae_bn3_pretrained = os.path.abspath(r"C:\Source\Research Repositories\TNBC\nets\CAE_bn3_001_pretrained.pt")
    # cae_bn3_pre_eval = "mods." + cae_bn3_name + "(img_size, n_clusters)"
    # cae_bn3_pre_model = eval(cae_bn3_pre_eval)
    # cae_bn3_pre_model = cae_bn3_pre_model.to(device)
    # cae_bn3_pre_model.load_state_dict(torch.load(cae_bn3_pretrained))
    # cae_bn3_pre_array, cae_bn3_pre_label_array, cae_bn3_pre_preds = trainers.calculate_predictions(cae_bn3_pre_model, val_loader, params)
    # nmi_cae_bn3_pre = utils.metrics.nmi(cae_bn3_pre_label_array, cae_bn3_preds)
    # ari_cae_bn3_pre = utils.metrics.ari(cae_bn3_pre_label_array, cae_bn3_preds)
    # acc_cae_bn3_pre = utils.metrics.acc(cae_bn3_pre_label_array, cae_bn3_preds)
    # plotting PCA on DCEC CAE_3 latrep
    # utils.PCA_plot(cae_bn3_pre_array, cae_bn3_pre_label_array, cae_bn3_pre_preds, cae_bn3_name, n_clusters) 
    # t-sne
    # utils.tsne_plot(cae_bn3_pre_array, cae_bn3_pre_label_array, cae_bn3_pre_preds, cae_bn3_name, n_clusters)

    # VAE
    # method = 'VAE2'
    # z = np.load(os.path.abspath(r"C:\Source\Research Repositories\TNBC\runs\VAE2_010\z.npy"))
    # labels = np.load(os.path.abspath(r"C:\Source\Research Repositories\TNBC\runs\VAE2_010\labels.npy"))
    # n_clusters = 10
    # kmeans = KMeans(init = 'k-means++',n_clusters=n_clusters).fit(z)
    # kmeans.fit(z)
    # nmi_vae = normalized_mutual_info_score(labels, kmeans.labels_)
    # ari_vae = utils.metrics.ari(labels, kmeans.labels_)
    # acc_vae = utils.metrics.acc(labels, kmeans.labels_)
    # # plots
    # utils.PCA_plot(z, labels, kmeans.labels_, method, n_clusters, n_components = 2)
    # utils.tsne_plot(z, labels, kmeans.labels_, method, n_clusters, n_components = 2)


    # VaDE
    method = 'VaDE'
    labels = np.load(os.path.abspath(r"C:\Source\Research Repositories\TNBC\runs\VaDE_001\labels_e99.npy"))
    preds = np.load(os.path.abspath(r"C:\Source\Research Repositories\TNBC\runs\VaDE_001\preds_e99.npy"))
    z = np.load(os.path.abspath(r"C:\Source\Research Repositories\TNBC\runs\VaDE_001\z_e99.npy"))
    n_clusters = 10
    kmeans = KMeans(init = 'k-means++',n_clusters=n_clusters).fit(z)
    kmeans.fit(z)
    nmi_vae = normalized_mutual_info_score(labels, kmeans.labels_)
    ari_vae = utils.metrics.ari(labels, kmeans.labels_)
    acc_vae = utils.metrics.acc(labels, kmeans.labels_)
    # plots
    utils.PCA_plot(z, labels, kmeans.labels_, method, n_clusters, n_components = 2)
    utils.tsne_plot(z, labels, kmeans.labels_, method, n_clusters, n_components = 2)


    # simple CAE
    model = mods.MnistCAE(None)
    model.load_state_dict(torch.load(r"C:\Source\Research Repositories\TNBC\MNIST\weights\weights_124.pth"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_batch_size = val_batch_size = 256
    train_loader, val_loader = datagen.get_data_loaders('MNIST-train', train_batch_size, val_batch_size)
    loss_function = nn.MSELoss()

    # optimizer, I've used Adadelta, as it wokrs well without any magic numbers
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    

    losses = []
    # progress bar (works in Jupyter notebook too!)
    progress = tqdm(enumerate(val_loader), desc="Loss: ", total=len(val_loader))
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
    nmi_cae = normalized_mutual_info_score(labels, km.labels_)
    ari_cae = utils.metrics.ari(labels, km.labels_)
    acc_cae = utils.metrics.acc(labels, km.labels_)

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
