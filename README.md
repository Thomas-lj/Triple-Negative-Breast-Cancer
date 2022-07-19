# Triple Negative Breast Cancer
This project seeks to identify tumor subtypes of triple negative breast cancer using deep unsupervised clustering. This project is part of Thomas' masters thesis, 2020.

# Deep Convolutional Embedded Clustering
The project implements Deep Convolutional Embedded Clustering (https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf) to optimize a joint objective function:

![image](https://user-images.githubusercontent.com/43189719/179661422-d78f70ec-8f0a-45fd-b721-32d659c4ac9b.png)


![image](https://user-images.githubusercontent.com/43189719/179661086-56ef5ec8-ecae-4f99-b53a-53f652e57944.png)


![image](https://user-images.githubusercontent.com/43189719/179660810-b8ddc64a-513c-4f78-82b6-36da06d4b65f.png)

# Tissues
A tissue classificator was implemented to cluster different tissue types: Tumor, stroma, fat and lymphocytes.
Example of cluster convergence with hyperparameters K=4 clusters, γ=0.1:
![Cluster training](https://user-images.githubusercontent.com/43189719/179666398-12fb2fab-9446-481b-829f-f9e2a70ccfa3.gif)

# Tumor
Tumor tiles like these were clustered assuming K=4, 8, 12 and 20 clusters.
Example of
![image](https://user-images.githubusercontent.com/43189719/179666798-894edc52-8d5e-44d5-9ca3-900f86df78c3.png)
Examples of Tumor tiles for K=4 clusters:
![image](https://user-images.githubusercontent.com/43189719/179667234-0e440021-afca-4785-8685-5ea7eafc3b2f.png)
