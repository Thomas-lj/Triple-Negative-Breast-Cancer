# Utilities function Thomas Leicht Jensen Sep. 2019

import sklearn.metrics
import torch
import os
import scipy.misc 
import numpy as np
import seaborn as sns
import tensorflow as tf
import scipy.cluster.hierarchy as shc
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import ImageDraw, ImageFont, Image 
from gap_statistic import OptimalK
from lifelines import KaplanMeierFitter, CoxPHFitter
from torchvision import transforms
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from torchvision.utils import save_image
from scipy.cluster.hierarchy import dendrogram, linkage

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
inv_normalize = transforms.Normalize(
    mean=[-0.485 / .229, -0.456 / 0.224, -0.406 / 0.255],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
)


# Simple tensor to image translation
def tensor2img(tensor):
    img = tensor.cpu().data[0]
    if img.shape[0] != 1:
        img = inv_normalize(img)
    img = torch.clamp(img, 0, 1)
    return img


# Define printing to console and file
def print_both(f, text):
    print(text)
    f.write(text + '\n')


# Metrics class was copied from DCEC article authors repository (link in README)
class metrics:
    nmi = sklearn.metrics.normalized_mutual_info_score
    ari = sklearn.metrics.adjusted_rand_score

    @staticmethod
    def acc(labels_true, labels_pred):
        labels_true = labels_true.astype(np.int64)
        assert labels_pred.size == labels_true.size
        D = max(labels_pred.max(), labels_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(labels_pred.size):
            w[labels_pred[i], labels_true[i]] += 1
        from sklearn.utils.linear_assignment_ import linear_assignment
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in ind]) * 1.0 / labels_pred.size

def sample_cluster(data, pred, save_path, n, name, labels=None):
    classes = np.unique(pred)
    img_pil = []
    for i in classes:
        im_pil = []
        np.random.seed(seed=71991)
        idx = np.random.choice(np.where(pred==i)[0], n)
        for k in idx:
            im2 = transforms.ToPILImage()(data[k][0])
            if labels is not None:
                lab = labels[k]
                font = ImageFont.truetype(r"C:\Windows\WinSxS\amd64_microsoft-windows-font-truetype-arial_31bf3856ad364e35_10.0.17134.1_none_5803fc87168579d6\arial.ttf", 20)  
                ImageDraw.Draw(im2).text((5,5), str(lab), fill=(0,0,0), font=font)
            im_pil.append(im2)
        pil_img = Image.fromarray(np.asarray(np.concatenate(im_pil, 1)))
        img_pil.append(pil_img)
    
    # img_pil = np.concatenate(img_pil)
    pil_img = Image.fromarray(np.concatenate(img_pil))
    pil_img.save(os.path.join(save_path, name + '.png'))

def sample_lowest_probs(loader, probs, pred, save_path, n, name, labels=None):
    classes = np.unique(pred)
    img_pil = []
    for i in classes:
        im_pil = []
        idx = np.where(pred==i)[0]
        vals = probs[idx,i]
        idx2 = idx[np.argsort(vals)][:n]
        for k in idx2:
            im2 = transforms.ToPILImage()(loader[k][0])
            if labels is not None:
                lab = labels[k]
                font = ImageFont.truetype(r"C:\Windows\WinSxS\amd64_microsoft-windows-font-truetype-arial_31bf3856ad364e35_10.0.17134.1_none_5803fc87168579d6\arial.ttf", 20)  
                ImageDraw.Draw(im2).text((5,5), str(lab), fill=(0,0,0), font=font)
            im_pil.append(im2)
        pil_img = Image.fromarray(np.asarray(np.concatenate(im_pil, 1)))
        img_pil.append(pil_img)
    
    pil_img = Image.fromarray(np.concatenate(img_pil))
    pil_img.save(os.path.join(save_path, name + '.png'))

def sample_highest_probs(loader, probs, pred, save_path, n, name, labels=None):
    classes = np.unique(pred)
    img_pil = []
    for i in classes:
        im_pil = []
        idx = np.where(pred==i)[0]
        vals = probs[idx,i]
        class_ordered = np.argsort(vals)
        idx2 = idx[class_ordered][-n:]
        # test = []
        #     for hej in idx2:
        #         test.append(loader.path[hej][75:90])
        for k in idx2:
            im2 = transforms.ToPILImage()(loader[k][0])
            if labels is not None:
                lab = labels[k]
                font = ImageFont.truetype(r"C:\Windows\WinSxS\amd64_microsoft-windows-font-truetype-arial_31bf3856ad364e35_10.0.17134.1_none_5803fc87168579d6\arial.ttf", 20)  
                ImageDraw.Draw(im2).text((5,5), str(lab), fill=(0,0,0), font=font)
            im_pil.append(im2)
        pil_img = Image.fromarray(np.asarray(np.concatenate(im_pil, 1)))
        img_pil.append(pil_img)
    
    pil_img = Image.fromarray(np.concatenate(img_pil))
    pil_img.save(os.path.join(save_path, name + '.png'))
    
def cluster_probs(loader, probs, preds, clust, compare_matrix, IDs, n):
    
    other_idx = np.where(preds!=clust)[0]
    other_ids = []
    for k in other_ids:
        other_ids.append(loader.path[k][75:90])
    clust_idx = np.where(preds==clust)[0]
    pos_ids = IDs[np.where(compare_matrix[:,0]==1)[0]]
    neg_ids = IDs[np.where(compare_matrix[:,1]==1)[0]]
    score_board = np.zeros((n, compare_matrix.shape[1]))
    for i in range(compare_matrix.shape[1]): 
        for k in range(n):
            print(k)
            
    return score_board

def PCA_plot(pca, labels, preds, x_lims, y_lims, save_path, method):
    targets = np.unique(labels)
    pred_labels = np.unique(preds)

    f = plt.figure(figsize=(13, 6))
    plt.subplot(1,2,1)
    for target in targets:
        idx = np.where(labels==target)
        plt.scatter(pca[idx[0], 0], pca[idx[0], 1], alpha=0.5)
    plt.xlabel('PC 1', fontsize=15)
    plt.ylabel('PC 2', fontsize=15)
    plt.xlim(x_lims[0]-0.25*abs(x_lims[0]), x_lims[1]+0.25*abs(x_lims[1]))
    plt.ylim(y_lims[0]-0.25*abs(y_lims[0]), y_lims[1]+0.25*abs(y_lims[1]))
    plt.title('True labels, for ' + method, fontsize=20)
    plt.legend(targets, fontsize='x-small')

    plt.subplot(1,2,2)
    for pred in pred_labels:
        idx2 = np.where(preds==pred)
        plt.scatter(pca[idx2[0], 0], pca[idx2[0], 1], alpha=0.5)
    plt.xlabel('PC 1', fontsize=15)
    plt.ylabel('PC 2', fontsize=15)
    plt.xlim(x_lims[0]-0.25*abs(x_lims[0]), x_lims[1]+0.25*abs(x_lims[1]))
    plt.ylim(y_lims[0]-0.25*abs(y_lims[0]), y_lims[1]+0.25*abs(y_lims[1]))
    plt.title('Predicted labels for ' + method, fontsize=20)
    plt.legend(pred_labels, fontsize = 'x-small')
    plt.savefig(os.path.join(save_path, method + '_PCA.png'))
    f.clear()
    plt.close(f)
    return

def TNBC_PCA(z, preds, x_lims, y_lims, save_path, method):
    f = plt.figure(figsize=(8, 8))
    plt.subplot(1,1,1)
    targets = np.unique(preds)
    for target in targets:
        idx = np.where(preds==target)
        plt.scatter(z[idx, 0], z[idx, 1], alpha=0.7)
    plt.legend(targets)
    plt.xlabel('PC 1', fontsize=15)
    plt.ylabel('PC 2', fontsize=15)
    plt.axis('equal')
    plt.xlim(x_lims[0]-0.25*abs(x_lims[0]), x_lims[1]+0.25*abs(x_lims[1]))
    plt.ylim(y_lims[0]-0.25*abs(y_lims[0]), y_lims[1]+0.25*abs(y_lims[1]))
    plt.title('PCA of ' + method, fontsize=20)
    plt.savefig(os.path.join(save_path, method + '.png'))
    f.clear()
    plt.close(f)
    return

def TNBC_tsne(tsne, preds, save_path, method):
    f = plt.figure(figsize=(8, 8))
    plt.subplot(1,1,1)
    targets = np.unique(preds)
    for target in targets:
        idx = np.where(preds==target)
        plt.scatter(tsne[idx, 0], tsne[idx, 1], alpha=0.7)
    plt.legend(targets)
    plt.xlabel('tsne 1', fontsize=15)
    plt.ylabel('tsne 2', fontsize=15)
    plt.axis('equal')
    plt.xlim(np.min(tsne[:,0])-0.25*abs(np.min(tsne[:,0])), np.max(tsne[:,0]) + 0.25*abs(np.max(tsne[:,0])))
    plt.ylim(np.min(tsne[:,1])-0.25*abs(np.min(tsne[:,1])), np.max(tsne[:,1]) + 0.25*abs(np.max(tsne[:,1])))
    plt.title('tsne of ' + method, fontsize=20)
    plt.savefig(os.path.join(save_path, method + '_tsne.png'))
    f.clear()
    plt.close(f)
    return 

def tsne_plot(t_space, labels, preds, save_path, method):
    targets = np.unique(labels)
    pred_labels = np.unique(preds)
    
    plt.figure(figsize=(13,6))
    plt.subplot(1,2,1)
    for target in targets:
        idx = np.where(labels==target)
        plt.scatter(t_space[idx[0], 0], t_space[idx[0], 1], alpha = 0.7)
    plt.xlabel('t-sne 1', fontsize=15)
    plt.ylabel('t-sne 2', fontsize=15)
    plt.title('True labels for ' + method, fontsize=20)
    plt.legend(targets, fontsize='x-small')

    plt.subplot(1,2,2)
    for pred in pred_labels:
        idx2 = np.where(preds==pred)
        plt.scatter(t_space[idx2, 0], t_space[idx2, 1], alpha = 0.7)
    plt.xlabel('t-sne 1', fontsize=15)
    plt.ylabel('t-sne 2', fontsize=15)
    plt.title('Predicted label for ' + method, fontsize=20)
    plt.legend(pred_labels, fontsize='x-small')
    plt.savefig(os.path.join(save_path, method + '_tsne.png'))
    return    

def plot_distributions(q, p, KL_div,  save_path):
    f = plt.figure(figsize=(13, 6))

    plt.subplot(1,2,1)
    for idx in range(q.shape[1]):
        sns.distplot(q[:,idx], hist=False, kde_kws={'linewidth': 3}, label=str(idx))
    plt.xlim((0, 1))
    plt.xlabel('q')
    plt.ylabel('Density')
    plt.legend(prop={'size': 10}, title = 'q')
    plt.title('q_ij, KL_div=' + str(KL_div))
    plt.subplot(1,2,2)
    for idx2 in range(p.shape[1]):
        sns.distplot(p[:,idx2], hist=False, kde=True, kde_kws={'linewidth': 3}, label=str(idx2))
    plt.xlim((0, 1))
    plt.xlabel('p')
    plt.ylabel('Density')
    plt.legend(prop={'size': 10}, title = 'p')
    plt.title('p_ij')
    plt.savefig(os.path.join(save_path + '.png'))
    f.clear()
    plt.close(f)
    return

def KM_fitter(surv_matrix, deaths, time_death, n, labels, save_path, method):
    bin_matrix = np.zeros((surv_matrix.shape))
    ax = plt.subplot(1,1,1)
    for column in range(surv_matrix.shape[1]):
        # ax = plt.subplot(1,1,1)
        # patient_idx = np.nonzero(surv_matrix[:,column])[0]
        patient_idx = np.where(surv_matrix[:,column] >= n)[0]
        bin_matrix[patient_idx, column] = 1
        d1 = deaths[patient_idx].values
        t1 = time_death[patient_idx].values
        kmf = KaplanMeierFitter() 
        kmf.fit(t1, d1)
        kmf.plot(ax=ax, ci_show=False, legend=False)
        plt.title('KM curve cluster assignments', fontsize=20)
        plt.xlim(0, 12)
        plt.ylim(0, 1)
        plt.legend(labels, loc=3)
        plt.ylabel('Fraction survival', fontsize=15)
        plt.xlabel('Time (years)', fontsize=15)
    plt.savefig(os.path.join(save_path, method + '_survivals.png'))
    return bin_matrix

def hazard_ratios(df, cph, save_path, save_name):
    # ax = plt.subplot(1,1,1)
    f = plt.figure(figsize=(13, 6))
    # cph.fit(df, duration_col='yearstodeath_TNBC', event_col='vitalstatus_TNBC')
    cph.plot()
    plt.title('Hazard ratios')
    plt.xlabel('log(HR) (95% CI)')
    plt.savefig(os.path.join(save_path, save_name + '.png'))
    f.clear()
    plt.close(f)
    return     
    
def get_heatmap(matrix, save_path, method, row_colors=None):
    f = plt.figure(figsize=(13, 6))
    g = sns.clustermap(matrix, row_colors=row_colors)
    g.ax_heatmap.set_title('Heatmap for K=' + str(matrix.shape[1]) + ' clusters.')
    plt.savefig(os.path.join(save_path, method + '_heatmap.png'))

def distr_pam50(pam_matrix, n_clusters, save_path, method):
    clust_labels = []
    for ch in np.arange(n_clusters):
        clust_labels.append(str(ch))
    width = 0.1
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(clust_labels))
    
    rects1 = ax.bar(x - 2*width, pam_matrix[:,0], width, label='Basal')
    rects2 = ax.bar(x - width, pam_matrix[:,1], width, label='Her2')
    rects3 = ax.bar(x, pam_matrix[:,2], width, label='LumA')
    rects4 = ax.bar(x + width, pam_matrix[:,3], width, label='LumB')
    rects5 = ax.bar(x + 2*width, pam_matrix[:,4], width, label='normal')

    ax.set_ylabel('Fraction of patients', fontsize=15)
    ax.set_xlabel('Cluster', fontsize=15)
    ax.set_title('PAM50 distributions of clusters', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(clust_labels)
    ax.legend(loc=1, fontsize='medium')
    plt.savefig(os.path.join(save_path, method + '_distribution.png'))

def distr_lehman(lehman_matrix, n_clusters, save_path, method):
    clust_labels = []
    for ch in np.arange(n_clusters):
        clust_labels.append(str(ch))
    width = 0.1
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(clust_labels))
    
    rects1 = ax.bar(x - 2*width, lehman_matrix[:,0], width, label='BL1')
    rects2 = ax.bar(x - width, lehman_matrix[:,1], width, label='BL2')
    rects3 = ax.bar(x, lehman_matrix[:,2], width, label='LAR')
    rects4 = ax.bar(x + width, lehman_matrix[:,3], width, label='M')
    
    ax.set_ylabel('Fraction of patients', fontsize=15)
    ax.set_xlabel('Cluster', fontsize=15)
    ax.set_title('Lehman distributions of clusters', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(clust_labels)
    ax.legend(loc=1, fontsize='medium')
    plt.savefig(os.path.join(save_path, method + '_distribution.png'))

def distr_T(T_matrix, n_clusters, save_path, method):
    clust_labels = []
    for ch in np.arange(n_clusters):
        clust_labels.append(str(ch))
    width = 0.1
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(clust_labels))
    np.sum(T_matrix[:,])
    T_matrix[:,0] = np.sum(T_matrix[:,[0,1]], 1)/np.sum(T_matrix,1)
    T_matrix[:,1] = T_matrix[:,2]/np.sum(T_matrix,1)
    T_matrix[:,2] = np.sum(T_matrix[:,[3, 4, 5, 6]], 1)/np.sum(T_matrix,1)
    T_matrix[:,3] = T_matrix[:,7]/np.sum(T_matrix,1) 
    rects1 = ax.bar(x - 2*width, T_matrix[:,0], width, label='T1')
    rects2 = ax.bar(x - width, T_matrix[:,1], width, label='T2')
    rects3 = ax.bar(x, T_matrix[:,2], width, label='T3+T4')
    rects4 = ax.bar(x + width, T_matrix[:,3], width, label='NA')
    
    ax.set_ylabel('Fraction of patients', fontsize=15)
    ax.set_xlabel('Cluster', fontsize=15)
    ax.set_title('Tumor stage distributions', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(clust_labels)
    ax.legend(loc=1, fontsize='medium')
    plt.savefig(os.path.join(save_path, method + '_distribution.png'))

def distr_stage(stage_matrix, n_clusters, save_path, method):
    clust_labels = []
    for ch in np.arange(n_clusters):
        clust_labels.append(str(ch))
    width = 0.1
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(clust_labels))
    
    stage_matrix[:,0] = np.sum(stage_matrix[:,[1, 2]], 1)/np.sum(stage_matrix,1)
    stage_matrix[:,1] = np.sum(stage_matrix[:,[3, 4]], 1)/np.sum(stage_matrix,1)
    stage_matrix[:,2] = np.sum(stage_matrix[:,[5, 6, 7, 8]], 1)/np.sum(stage_matrix,1)
    stage_matrix[:,3] = np.sum(stage_matrix[:,[0, 9]], 1)/np.sum(stage_matrix,1)

    rects1 = ax.bar(x - 2*width, stage_matrix[:,0], width, label='I')
    rects2 = ax.bar(x - width, stage_matrix[:,1], width, label='II')
    rects3 = ax.bar(x, stage_matrix[:,2], width, label='III+IV')
    rects4 = ax.bar(x + width, stage_matrix[:,3], width, label='NA')
    
    ax.set_ylabel('Fraction of patients', fontsize=15)
    ax.set_xlabel('Cluster', fontsize=15)
    ax.set_title('Overall disease stage', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(clust_labels)
    ax.legend(loc=1, fontsize='medium')
    plt.savefig(os.path.join(save_path, method + '_distribution.png'))

def findK(data, n_clusters):
    optimalK = OptimalK(parallel_backend='rust')
    n_clusters = optimalK(data, cluster_array=np.arange(1, n_clusters))
    print('Optimal clusters: ', n_clusters)
    
    plt.plot(optimalK.gap_df.n_clusters, optimalK.gap_df.gap_value, linewidth=3)
    plt.scatter(optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].n_clusters,
                optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].gap_value, s=250, c='r')
    plt.grid(True)
    plt.xlabel('Cluster Count')
    plt.ylabel('Gap Value')
    plt.title('Gap Values by Cluster Count')
    return plt.show()

def dendrofile(data):
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        data,
        leaf_rotation=180.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    return plt.show()
