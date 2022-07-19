# Triple Negative Breast Cancer
This project seeks to identify tumor subtypes of triple negative breast cancer using deep unsupervised clustering. This project is part of a masters thesis, 2020.

# Deep Convolutional Embedded Clustering
The project implements Deep Convolutional Embedded Clustering (https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf) to optimize a joint objective function:

![image](https://user-images.githubusercontent.com/43189719/179661422-d78f70ec-8f0a-45fd-b721-32d659c4ac9b.png)


![image](https://user-images.githubusercontent.com/43189719/179661086-56ef5ec8-ecae-4f99-b53a-53f652e57944.png)


![image](https://user-images.githubusercontent.com/43189719/179660810-b8ddc64a-513c-4f78-82b6-36da06d4b65f.png)

# Tissues
As part of a proof of concept, a tissue classificator was implemented to cluster different tissue types.
Convergence with K=4 clusters, γ=0.1>
![Picture1](https://user-images.githubusercontent.com/43189719/179665649-3ad1166e-9f2c-4ec9-9b82-a74ef47fbfdb.gif)
![](http://i.imgur.com/60bts.gif)


# Tumor
Tumor tiles like these were clustered assuming K=4, 8, 12 and 20 clusters.
![image](https://user-images.githubusercontent.com/43189719/179660585-3a192ed3-9ac7-4f92-a701-524934511528.png)

# Cluster training
Example of clustering during training.

![image](https://user-images.githubusercontent.com/43189719/179664975-5c2c682c-f4a2-4404-aa1c-79264c1fb007.png)
