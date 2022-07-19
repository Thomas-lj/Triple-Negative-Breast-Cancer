# Triple Negative Breast Cancer
This project seeks to identify tumor subtypes of triple negative breast cancer using deep unsupervised clustering. This project is part of Thomas' masters thesis, 2020.

# Deep Convolutional Embedded Clustering
The project implements Deep Convolutional Embedded Clustering (https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf) to optimize a joint objective function (L) with both a reconstruction (L_r) and clustering loss (L_c):

Reconstruction loss:

![image](https://user-images.githubusercontent.com/43189719/179667511-46878b89-9730-4777-aaef-9e285e41a87b.png)

Clustering loss:

![image](https://user-images.githubusercontent.com/43189719/179667553-7952a321-855e-439e-bb37-dfc60922b1fc.png)

Joint optimization:

![image](https://user-images.githubusercontent.com/43189719/179667594-a8287443-e19f-4ea9-97b6-94ddca276eda.png)

Model presentation (as illustrated by Guo. et al.) of DCEC, where q is the latent representation vector of an image:
![image](https://user-images.githubusercontent.com/43189719/179660810-b8ddc64a-513c-4f78-82b6-36da06d4b65f.png)

# Tissues
A tissue classificator was implemented to cluster different tissue types: Tumor, stroma, fat and lymphocytes.
Examples of tissue types, from left column: fat, stroma, lymphocytes and tumor:

![image](https://user-images.githubusercontent.com/43189719/179668197-d6265480-a9ab-4f67-81f6-976c9d68925c.png)

This illustrates Principal Component (PC1 and PC2 on the axes) from the latent representation q (the condensed image presentation between the encoder and decoder).
Example of cluster convergence with hyperparameters K=4 clusters, γ=0.1:

![Cluster training](https://user-images.githubusercontent.com/43189719/179666398-12fb2fab-9446-481b-829f-f9e2a70ccfa3.gif)

# Tumor
Tumor tiles like these were clustered assuming K=4, 8, 12 and 20 clusters.
Example of cluster assignment latent representations:

![image](https://user-images.githubusercontent.com/43189719/179666798-894edc52-8d5e-44d5-9ca3-900f86df78c3.png)

Examples of Tumor tiles for K=4 clusters suggests a color dependency to the cluster assignments:

![image](https://user-images.githubusercontent.com/43189719/179667234-0e440021-afca-4785-8685-5ea7eafc3b2f.png)
