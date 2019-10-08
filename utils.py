# Utilities function Thomas Leicht Jensen Sep. 2019
import numpy as np
import sklearn.metrics
import torch
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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


def evaluation(val_loader, model_name, weights, batch_size, loss_function):
        # get device (cuda if avail.)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # progress bar
        val_batches = len(val_loader)
        progress = tqdm(enumerate(val_loader), desc="Loss: ", total=val_batches)
        model = mods.model_name(None)
        labels_train = []
        z_train = []
        model.load_state_dict(torch.load(weights))
        
        for i, data in progress:
            X, y = data[0].to(device), data[1].to(device)
            model.zero_grad()
            emb = model.embed(X)
            # np.concatenate(y.cpu().numpy(), labels_train)
            labels_train.append(y.cpu().numpy())
            z_train.append(emb.detach().cpu().numpy())

        labels_train = np.array(labels_train)
        labels = np.concatenate(labels_train)
        z_train = np.array(z_train)
        z = np.ndarray.squeeze(np.concatenate(z_train))
        return labels, z

def PCA_plot(z, labels, preds, method, n_clusters = 10, n_components = 2):
    targets = np.unique(labels)
    num_targets = len(labels)
    clusters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    clusters = clusters[0:n_clusters]
    pca = PCA(n_components).fit_transform(z)
    
    plt.subplot(2,1,1)
    for target in targets:
        idx = np.where(labels==target)
        plt.scatter(pca[idx[0], 0], pca[idx[0], 1])
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('PCA of MNIST with true labels')
    plt.legend(targets)

    plt.subplot(2,1,2)

    for target in targets:
        idx2 = np.where(preds==target)
        plt.scatter(pca[idx2[0], 0], pca[idx2[0], 1])
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('PCA of Mnist with ' + method)
    plt.legend(clusters)
    return plt.show()

def tsne_plot(z, labels, preds, method, n_clusters = 10 ,n_components = 2):
    targets = np.unique(labels)
    clusters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    clusters = clusters[0:n_clusters]
    tsne = TSNE(n_components = 2)
    t_space = tsne.fit_transform(z)

    plt.subplot(2,1,1)
    for target in targets:
        idx = np.where(labels==target)
        plt.scatter(t_space[idx[0], 0], t_space[idx[0], 1])
    plt.xlabel('t-sne 1')
    plt.ylabel('t-sne 2')
    plt.title('t-sne of MNIST with true labels')
    plt.legend(targets)

    plt.subplot(2,1,2)

    for target in targets:
        idx2 = np.where(preds==target)
        plt.scatter(t_space[idx2[0], 0], t_space[idx2[0], 1])
    plt.xlabel('t-sne 1')
    plt.ylabel('t-sne 2')
    plt.title('t-sne of Mnist with ' + method)
    plt.legend(clusters)
    return plt.show()