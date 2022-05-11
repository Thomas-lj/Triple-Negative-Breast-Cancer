# Evaluation script, Thomas
if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(r'C:\Source\Research Repositories\TNBC'))
    from datagen import get_data_loaders

    import inspect
    import time
    import torch
    import argparse
    import csv
    import seaborn as sns
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    import pandas as pd
    from collections import Counter
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.utils import k_fold_cross_validation
    from lifelines.statistics import logrank_test
    import statsmodels.api as smf
    from torchvision.datasets import MNIST
    from tqdm.autonotebook import tqdm
    from torch.utils import data
    from sklearn import preprocessing
    from sklearn.metrics import normalized_mutual_info_score
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import confusion_matrix as cmx
    from sklearn.metrics import silhouette_score as SHS
    from sklearn.metrics import davies_bouldin_score as DBI 
    from sklearn.metrics import calinski_harabasz_score as CHI
    from sklearn.preprocessing import normalize
    from torch import nn, optim
    from gap_statistic import OptimalK
    from torchvision.utils import save_image
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.stats.distributions import chi2

    import mods
    import datagen
    import trainers
    import utils
    seed = 71991

    # baseline naive kmeans for tissues, don't edit this part
    # Tissues (fat, stroma, lymphocytes and tumor) for CAE_3, naive kmeans

    # model_name = 'TNBC_CAE_3'
    # weights_path = os.path.abspath(r"C:\Source\Research Repositories\TNBC\nets\TissuesTNBC_CAE_3_001_pretrained.pt")
    # save_path = os.path.abspath(r"C:\Source\Research Repositories\TNBC\runs\TissuesTNBC_CAE_3_001")
    # train_path = os.path.abspath(r"Z:\Speciale\Tissues\Train")
    # val_path = os.path.abspath(r"Z:\Speciale\Tissues\Eval")
    # train_batch_size = val_batch_size = 32
    # transforms = transforms.Compose([transforms.CenterCrop(128),
    #                                     transforms.RandomHorizontalFlip(),
    #                                      transforms.ToTensor()])
    # train = datagen.DataGen_Tissues(train_path, transform=transforms)
    # val = datagen.DataGen_Tissues(val_path, transform=transforms)
    # train_loader = data.DataLoader(train, batch_size=train_batch_size, shuffle=False)
    # val_loader = data.DataLoader(val, batch_size=val_batch_size, shuffle=False)
    # model_name = 'TNBC_CAE_3'
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # params = {'device': device}
    # img_size = [3, 128, 128]
    # n_clusters = 4
    # z_dim = 512
    # seed = 71991
    # to_eval = "mods." + model_name + "(img_size, num_clusters=n_clusters, z_dim=z_dim)"
    # model = eval(to_eval)
    # model = model.to(device)
    # model.load_state_dict(torch.load(weights_path))
    # x_tilde, x, output_array, z, label_array, _ = trainers.calculate_predictions(model, train_loader)
    # x_tilde2, x2, output_array2, z2, label_array2, _ = trainers.calculate_predictions(model, val_loader)
    
    # km = KMeans(n_clusters, n_init=20).fit_predict(z)
    # km2 = KMeans(n_clusters, n_init=20).fit_predict(z2)

    # # sample cluster images
    # utils.sample_cluster(train_loader.dataset, km, save_path, 10, 'Tissues_cluster_train.png')
    # utils.sample_cluster(val_loader.dataset, km2, save_path, 10, 'Tissues_cluster_eval.png')
    
    # # metrics
    # nmi_train = utils.metrics.nmi(label_array, km, average_method='arithmetic')
    # ari_train = utils.metrics.ari(label_array, km)
    # acc_train = utils.metrics.acc(label_array, km)
    # nmi_eval = utils.metrics.nmi(label_array2, km2, average_method='arithmetic')
    # ari_eval = utils.metrics.ari(label_array2, km2)
    # acc_eval = utils.metrics.acc(label_array2, km2)
    # cm_train = cmx(label_array, km)
    # cm_eval = cmx(label_array2, km2)

    # np.savetxt(os.path.join(save_path, 'cm_train.csv'), cm_train, fmt='%10.5f')
    # np.savetxt(os.path.join(save_path, 'cm_eval.csv'), cm_eval, fmt='%10.5f')
    # data = {'NMI_train': nmi_train, 'ARI_train': ari_train, 'ACC_train': acc_train, 
    #         'NMI_eval': nmi_eval, 'ARI_eval': ari_eval, 'ACC_eval': acc_eval 
    #         }
    # with open(os.path.join(save_path, 'metrics.csv'), 'w') as f:
    #     w = csv.DictWriter(f, data.keys())
    #     w.writeheader()
    #     w.writerow(data)
    
    # pca = PCA(2, random_state=seed).fit(z)
    # train_pca = pca.transform(z)
    # eval_pca = pca.transform(z2)

    # utils.PCA_plot(train_pca, label_array, km, save_path, 'Tissues_train')
    # utils.PCA_plot(eval_pca, label_array2, km2, save_path, 'Tissues_eval')

    # # Fit tsne to 2000 images from training
    # tsne_train = TSNE(n_components=2, random_state=71991).fit_transform(z)
    # tsne_eval = TSNE(n_components=2, random_state=71991).fit_transform(z2)
    # utils.tsne_plot(tsne_train, label_array, km, save_path, 'Tissues_train')
    # utils.tsne_plot(tsne_eval, label_array2, km2, save_path, 'Tissues_eval')

# --------------------------------------------------------------------------------------------------------------------------------------
    # DCEC Tissues CAE_3
    # model_name = 'TNBC_CAE_3'
    # weights_path = os.path.abspath(r"C:\Source\Research Repositories\TNBC\nets\Tissues_TNBC_CAE_3_005.pt")
    # save_path = os.path.abspath(r"C:\Source\Research Repositories\TNBC\runs\Tissues_TNBC_CAE_3_005")
    # train_path = os.path.abspath(r"Z:\Speciale\Tissues\Train")
    # val_path = os.path.abspath(r"Z:\Speciale\Tissues\Eval")
    # train_batch_size = val_batch_size = 32
    # transforms = transforms.Compose([transforms.CenterCrop(128),
    #                                     transforms.RandomHorizontalFlip(),
    #                                      transforms.ToTensor()])
    # train = datagen.DataGen_Tissues(train_path, transform=transforms)
    # val = datagen.DataGen_Tissues(val_path, transform=transforms)
    # train_loader = data.DataLoader(train, batch_size=train_batch_size, shuffle=False)
    # val_loader = data.DataLoader(val, batch_size=val_batch_size, shuffle=False)
    # model_name = 'TNBC_CAE_3'
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # params = {'device': device}
    # img_size = [3, 128, 128]
    # n_clusters = 4
    # z_dim = 512
    # seed = 71991
    # to_eval = "mods." + model_name + "(img_size, num_clusters=n_clusters, z_dim=z_dim)"
    # model = eval(to_eval)
    # model = model.to(device)
    # model.load_state_dict(torch.load(weights_path))
    # x_tilde, x, output_array, z, label_array, preds = trainers.calculate_predictions(model, train_loader)
    # x_tilde2, x2, output_array2, z2, label_array2, preds2 = trainers.calculate_predictions(model, val_loader)
    
    # sample cluster images
    # utils.sample_cluster(train_loader.dataset, preds, save_path, 10, 'Tissues_cluster_train', labels=label_array)
    # utils.sample_cluster(val_loader.dataset, preds2, save_path, 10, 'Tissues_cluster_eval', labels=label_array2)
    
    # metrics
    # nmi_train = utils.metrics.nmi(label_array, preds, average_method='arithmetic')
    # ari_train = utils.metrics.ari(label_array, preds)
    # acc_train = utils.metrics.acc(label_array, preds)
    # nmi_eval = utils.metrics.nmi(label_array2, preds2, average_method='arithmetic')
    # ari_eval = utils.metrics.ari(label_array2, preds2)
    # acc_eval = utils.metrics.acc(label_array2, preds2)
    # cm_train = cmx(label_array, preds)
    # cm_eval = cmx(label_array2, preds2)
    # chi_train = CHI(z, preds)
    # chi_eval = CHI(z2, preds2)

    # np.savetxt(os.path.join(save_path, 'cm_train.csv'), cm_train, fmt='%10.5f')
    # np.savetxt(os.path.join(save_path, 'cm_eval.csv'), cm_eval, fmt='%10.5f')
    # data = {'NMI_train': nmi_train, 'ARI_train': ari_train, 'ACC_train': acc_train, 
    #         'NMI_eval': nmi_eval, 'ARI_eval': ari_eval, 'ACC_eval': acc_eval, 'CHI_train': chi_train, 'CHI_eval': chi_eval  
    #         }
    # with open(os.path.join(save_path, 'metrics.csv'), 'w') as f:
    #     w = csv.DictWriter(f, data.keys())
    #     w.writeheader()
    #     w.writerow(data)
    
    # pca = PCA(2, random_state=seed).fit(z)
    # pca_train = pca.transform(z)
    # pca_eval = pca.transform(z2)
    # x_lim_t = (np.min(pca_train[:,0]), np.max(pca_train[:,0]))
    # y_lim_t = (np.min(pca_train[:,1]), np.max(pca_train[:,1]))
    # x_lim_e = (np.min(pca_eval[:,0]), np.max(pca_eval[:,0]))
    # y_lim_e = (np.min(pca_eval[:,1]), np.max(pca_eval[:,1]))
    # utils.PCA_plot(pca_train, label_array, preds, x_lim_t, y_lim_t, save_path, 'Tissues_train')
    # utils.PCA_plot(pca_eval, label_array2, preds2, x_lim_e, y_lim_e, save_path, 'Tissues_eval')

    # tsne_train = TSNE(n_components=2, random_state=71991).fit_transform(z)
    # tsne_eval = TSNE(n_components=2, random_state=71991).fit_transform(z2)
    # utils.tsne_plot(tsne_train, label_array, preds, save_path, 'Tissues_train')
    # utils.tsne_plot(tsne_eval, label_array2, preds2, save_path, 'Tissues_eval')

# survival curves
    # meta_path = os.path.abspath(r"C:\Source\Research Repositories\TNBC\data\GDC_metadata_exclusive.xlsx")
    # df = pd.read_excel(meta_path, header=[1,])
    # IDs = df.TCGA_SAMPLE
    # deaths = df.vitalstatus_TNBC
    # time_death = df.yearstodeath_TNBC
    # survival_matrix = np.zeros((len(IDs), len(np.unique(preds))))
    # survival_matrix = np.zeros((len(IDs), n_clusters))
    # for i, paths in enumerate(train_loader.dataset.path):
    #     patient_idx = np.where(IDs==paths.split('\\')[-2][36:51])[0][0]
    #     clust = preds[i]
    #     survival_matrix[patient_idx, clust] += 1
    
    # remove columns where no clusters are aparent
    # survival_matrix = np.delete(survival_matrix, np.where(np.sum(survival_matrix, axis=0)==0)[0], axis=1)
    # cols = ['PAM50lite', 'TNBCtype_4', 'neoplasm.diseasestage', 'pathology.T.stage', 'pathology.N.stage', 'pathology.M.stage', 'yearstodeath_TNBC', 'vitalstatus_TNBC']
    # df_dummies = pd.get_dummies(df[cols])
    # bin_matrix = utils.KM_fitter(survival_matrix, deaths, time_death, save_path)
    
    # df_all = pd.concat([df_dummies, pd.DataFrame(bin_matrix)], axis=1, sort=False)
    # df_clust = pd.concat([pd.DataFrame(bin_matrix), df['yearstodeath_TNBC'], df['vitalstatus_TNBC']], axis=1)
    # utils.hazard_ratios(df_clust, save_path, penalizer=0.2)

#---------------------------------------------------------------------------------------------------------------------------------------


# DCEC Tumors CAE_3
    model_name = 'TNBC_CAE_3'
    weights_path = os.path.abspath(r"Z:\Speciale\Results\Results\Vary_gamma\Tumor\K4\Tumor_TNBC_CAE_3_gamma_0.6\Tumor_TNBC_CAE_3_002.pt")
    save_path = os.path.abspath(r"C:\Source\Research Repositories\TNBC\runs\Other_cluster")
    train_path = os.path.abspath(r"Z:\Speciale\Tumor_other_clusters")
    train_batch_size = val_batch_size = 256
    transforms = transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()])
    train = datagen.DataGen_Tumor(train_path, 400, transform=transforms)
    train_loader = data.DataLoader(train, batch_size=train_batch_size, shuffle=False)
    model_name = 'TNBC_CAE_3'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params = {'device': device}
    img_size = [3, 128, 128]
    n_clusters = 4
    z_dim = 512
    seed = 71991
    to_eval = "mods." + model_name + "(img_size, num_clusters=n_clusters, z_dim=z_dim)"
    model = eval(to_eval)
    model = model.to(device)
    model.load_state_dict(torch.load(weights_path))
    x_tilde, x, output_array, z, label_array, preds = trainers.calculate_predictions(model, train_loader)
    
    # sample cluster images
    utils.sample_cluster(train_loader.dataset, preds, save_path, 20, 'Tissues_cluster_train')
    utils.sample_highest_probs(train_loader.dataset, output_array, preds, save_path, 10, 'Tissues_highest_prob')
    utils.sample_lowest_probs(train_loader.dataset, output_array, preds, save_path, 10, 'Tissues_lowest_prob')
    chi_score = CHI(z, preds)
    silhouette_score = SHS(z, preds)
    dbi_score = DBI(z, preds)
    
    data = {'CHI': chi_score, 'SSE': silhouette_score, 'DBI': dbi_score}
    with open(os.path.join(save_path, 'internal_metrics.csv'), 'w') as f:
        w = csv.DictWriter(f, data.keys())
        w.writeheader()
        w.writerow(data)
    
    pca = PCA(2, random_state=seed).fit(z)
    pca_train = pca.transform(z)
    x_lim = (np.min(pca_train[:,0]), np.max(pca_train[:,0]))
    y_lim = (np.min(pca_train[:,1]), np.max(pca_train[:,1]))
    utils.TNBC_PCA(pca_train, preds, x_lim, y_lim, save_path, 'Tumor')

    np.random.seed(seed=71991)
    idx_train = np.random.choice(len(label_array),2000,replace=False)
    tsne_train = TSNE(n_components=2, random_state=71991).fit_transform(z)
    utils.TNBC_tsne(tsne_train, preds, save_path, 'Tumor')

    # survival curves
    meta_path = os.path.abspath(r"C:\Source\Research Repositories\TNBC\data\GDC_metadata_exclusive.xlsx")
    df = pd.read_excel(meta_path, header=[1,])
    IDs = df.TCGA_SAMPLE
    deaths = df.vitalstatus_TNBC
    time_death = df.yearstodeath_TNBC
    survival_matrix = np.zeros((len(IDs), n_clusters))
    for i, paths in enumerate(train_loader.dataset.path):
        patient_idx = np.where(IDs==paths.split('\\')[-2][36:51])[0][0]
        clust = preds[i]
        survival_matrix[patient_idx, clust] += 1
    
    # remove columns where no clusters are aparent
    survival_matrix = np.delete(survival_matrix, np.where(np.sum(survival_matrix, axis=0)==0)[0], axis=1)

    pam_types = np.unique(df['PAM50'])
    lehman_types = np.unique(df['TNBCtype_4'])
    tumor_types = np.unique(df['pathology.T.stage'])
    stage_types = np.unique(df['neoplasm.diseasestage'].astype(str))
    pam_matrix = np.zeros((n_clusters, len(pam_types)))
    lehman_matrix = np.zeros((n_clusters, len(lehman_types)))
    tumor_matrix = np.zeros((n_clusters, len(tumor_types)))
    stage_matrix = np.zeros((n_clusters, len(stage_types)))


    for k in range(n_clusters):
        for i, j in enumerate(pam_types):
            idx = np.where(df['PAM50']==j)[0]
            pam_matrix[k, i] = np.sum(survival_matrix[idx, k])/np.sum(survival_matrix[:, k])
        for l, m in enumerate(lehman_types):
            idx2 = np.where(df['TNBCtype_4']==m)[0]
            lehman_matrix[k, l] = np.sum(survival_matrix[idx2, k])/np.sum(survival_matrix[:, k])
        for n, o in enumerate(tumor_types):
            idx3 = np.where(df['pathology.T.stage']==o)[0]
            tumor_matrix[k, n] =  np.sum(survival_matrix[idx3, k])
        for p, q in enumerate(stage_types):
            idx4 = np.where(df['neoplasm.diseasestage']==q)[0]
            stage_matrix[k, p] =  np.sum(survival_matrix[idx4, k])
   
    # cluster distributions
    utils.distr_pam50(pam_matrix, n_clusters, save_path, 'PAM50')
    utils.distr_lehman(lehman_matrix, n_clusters, save_path, 'Lehman')
    utils.distr_T(tumor_matrix, n_clusters, save_path, 'T_stage')
    utils.distr_stage(stage_matrix, n_clusters, save_path, 'Disease_stage')
    # lehman = dict(zip(df['TNBCtype_4'].unique(), "rbgy"))
    # pam = dict(zip(df['PAM50lite'].unique(), "rb"))
    # lehman_colors = df['TNBCtype_4'].map(lehman)
    # pam_colors = df['PAM50lite'].map(pam)
    # utils.get_heatmap(pd.DataFrame(norm_row_matrix), save_path, 'Lehman', lehman_colors)
    # utils.get_heatmap(pd.DataFrame(norm_row_matrix), save_path, 'PAM50', pam_colors)


    norm_col_matrix = survival_matrix / survival_matrix.max(axis=0) #normalized by clusters
    norm_row_matrix = normalize(survival_matrix, axis=1, norm='l1') # normalized by samples
    arg_max_matrix = np.zeros((survival_matrix.shape))
    idx_max = np.argmax(survival_matrix, axis=1)
    for i, k in enumerate(idx_max):
        arg_max_matrix[i, k] = 1
    
    cols = ['PAM50lite', 'PAM50', 'TNBCtype_4']
    df_pam50_lite = pd.concat([pd.get_dummies(df[cols[0]]), df['yearstodeath_TNBC'], df['vitalstatus_TNBC']], axis=1)
    df_pam50 = pd.concat([pd.get_dummies(df[cols[1]]), df['yearstodeath_TNBC'], df['vitalstatus_TNBC']], axis=1)
    df_lehman = pd.concat([pd.get_dummies(df[cols[2]]), df['yearstodeath_TNBC'], df['vitalstatus_TNBC']], axis=1)
    bin5_matrix = utils.KM_fitter(survival_matrix, deaths, time_death, 5, range(survival_matrix.shape[1]), save_path, 'sens=5_KM')

    # clusters
    df_clust = pd.concat([pd.DataFrame(bin5_matrix), df['yearstodeath_TNBC'], df['vitalstatus_TNBC']], axis=1)
    df_con = pd.concat([pd.DataFrame(norm_row_matrix), df['yearstodeath_TNBC'], df['vitalstatus_TNBC']], axis=1)
    df_col = pd.concat([pd.DataFrame(norm_col_matrix), df['yearstodeath_TNBC'], df['vitalstatus_TNBC']], axis=1)
    df_bin = pd.concat([pd.DataFrame(arg_max_matrix), df['yearstodeath_TNBC'], df['vitalstatus_TNBC']], axis=1)
    

    # fit binary, doesn't work for K=8 for gamma=1
    cph_bin = CoxPHFitter(penalizer=0.1)
    cph_bin.fit(df_bin, duration_col='yearstodeath_TNBC', event_col='vitalstatus_TNBC', step_size=0.5)
    cph_bin.print_summary()
    utils.hazard_ratios(df_bin, cph_bin, save_path, 'Hazard_ratio_arg_max')
    bin_matrix = utils.KM_fitter(arg_max_matrix, deaths, time_death, 1, range(arg_max_matrix.shape[1]), save_path, 'arg_max_KM')

    # fit continuous Cox
    cph_con = CoxPHFitter(penalizer=0.1)
    cph_con.fit(df_con, duration_col='yearstodeath_TNBC', event_col='vitalstatus_TNBC', step_size=0.5)
    # cph_con.print_summary()
    utils.hazard_ratios(df_con, cph_con, save_path, 'Hazard_ratio_continuous')

    #fit col norm
    cph_col = CoxPHFitter(penalizer=0)
    cph_col.fit(df_col, duration_col='yearstodeath_TNBC', event_col='vitalstatus_TNBC', step_size=0.5)
    utils.hazard_ratios(df_col, cph_col, save_path, 'Hazard_ratio_column')

    # fit 
    cph = CoxPHFitter(penalizer=0)
    cph.fit(df_clust, duration_col='yearstodeath_TNBC', event_col='vitalstatus_TNBC', step_size=0.5)
    cph.check_assumptions(df_clust)
    # cph.print_summary()
    utils.hazard_ratios(df_clust, cph, save_path, 'Hazard_ratio_clusters')
    # # fit only significant cluster 0:
    # df_0 = pd.concat([pd.DataFrame(bin_matrix[:,0]), df['yearstodeath_TNBC'], df['vitalstatus_TNBC']], axis=1)
    # cph0 = CoxPHFitter(penalizer=0)
    # cph0.fit(df_0, duration_col='yearstodeath_TNBC', event_col='vitalstatus_TNBC', step_size=0.5)
    alpha = 0.05
    for clust, p in enumerate(cph._compute_p_values()):
        if p<alpha:
            compare_matrix = np.zeros((bin5_matrix.shape[0], 2))
            compare_matrix[:,0] = bin5_matrix[:,clust]
            for idx in range(bin5_matrix.shape[0]):
                if compare_matrix[idx,clust]!=1:
                    compare_matrix[idx, 1] = 1
            print('Cluster: %s is significant, HR=%s' % (clust, cph.params_[clust]))
            # logrank test, hypothesis test two survival functions:
            T1 = time_death[compare_matrix[:,0]==1]
            T2 = time_death[compare_matrix[:,1]==1]
            E1 = deaths[compare_matrix[:,0]==1]
            E2 = deaths[compare_matrix[:,1]==1]
            results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
            kmf = KaplanMeierFitter()
            ax = plt.subplot(111)
            ax = kmf.fit(T1, E1).plot(ax=ax, ci_show=False, legend=False)
            ax = kmf.fit(T2, E2).plot(ax=ax, ci_show=False, legend=False)
            plt.title('KM survival curves', fontsize=20)
            plt.xlim(0, 12)
            plt.ylim(0, 1)
            plt.legend(('Cluster '+str(clust), 'Others'), loc=3)
            ax.text(0.9, 0.9, 'p='+str(round(results.p_value, 4)), transform=plt.gca().transAxes, fontsize=14, ha='center')
            plt.ylabel('Fraction survival', fontsize=15)
            plt.xlabel('Time (years)', fontsize=15)
            plt.savefig(os.path.join(save_path,  'Clust_'+str(clust)+'_vs_all_survivals.png'))

            # cluster negative group
            score_board = utils.cluster_probs(train_loader.dataset, output_array, preds, clust, compare_matrix, IDs, 200)
            neg_ids = IDs[np.where(compare_matrix[:,1]==1)[0]]
            pos_ids = IDs[np.where(compare_matrix[:,0]==1)[0]]


            df_uni = pd.concat([pd.DataFrame(bin5_matrix[:, clust]), df['yearstodeath_TNBC'], df['vitalstatus_TNBC']], axis=1)
            cph_uni = CoxPHFitter(penalizer=0)
            cph_uni.fit(df_uni, duration_col='yearstodeath_TNBC', event_col='vitalstatus_TNBC', step_size=0.5)
            LR = 2*(cph.log_likelihood_ - cph_uni.log_likelihood_)
            p2 = chi2.sf(LR, survival_matrix.shape[0])
            print('log-likelihood p-value:' + str(p2))
            cph_uni.print_summary()

    

    
    cph_lehman = CoxPHFitter(penalizer=0.1)
    cph_lehman.fit(df_lehman, duration_col='yearstodeath_TNBC', event_col='vitalstatus_TNBC', step_size=0.5)
    
    cph_pam50_lite = CoxPHFitter(penalizer=0.1)
    cph_pam50_lite.fit(df_pam50_lite, duration_col='yearstodeath_TNBC', event_col='vitalstatus_TNBC', step_size=0.5)
    
    cph_pam50 = CoxPHFitter(penalizer=0.1)
    cph_pam50.fit(df_pam50, duration_col='yearstodeath_TNBC', event_col='vitalstatus_TNBC', step_size=0.5)
    
    utils.hazard_ratios(df_lehman, cph_lehman, save_path, 'Hazard_ratio_Lehman')
    utils.hazard_ratios(df_pam50_lite, cph_pam50_lite, save_path, 'Hazard_ratio_PAM50_lite')
    utils.hazard_ratios(df_pam50, cph_pam50, save_path, 'Hazard_ratio_PAM50')

    
    
#----------------------------------------------------------------------------------------------------------------------------------------
    # DCEC CAE_3, TNBC
    train_path = os.path.abspath(r"Z:\Speciale\Tissues\Train")
    val_path = os.path.abspath(r"Z:\Speciale\Tissues\Eval")
    weights_path = os.path.abspath(r"C:\Source\Research Repositories\TNBC\nets\TissuesTNBC_CAE_3_002.pt")
    save_path = os.path.abspath(r"C:\Source\Research Repositories\TNBC\runs\TissuesTNBC_CAE_3_002")
    train_batch_size = val_batch_size = 32
    transforms = transforms.Compose([transforms.CenterCrop(128),
                                        transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor()])
    train = datagen.DataGen_Tissues(train_path, transform=transforms)
    val = datagen.DataGen_Tissues(val_path, transform=transforms)
    train_loader = data.DataLoader(train, batch_size=train_batch_size, shuffle=False)
    val_loader = data.DataLoader(val, batch_size=val_batch_size, shuffle=False)
    model_name = 'TNBC_CAE_3'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_size = [3, 128, 128]
    n_clusters = 4
    to_eval = "mods." + model_name + "(img_size, num_clusters=n_clusters)"
    model = eval(to_eval)
    model = model.to(device)
    model.load_state_dict(torch.load(weights_path))
    x_tilde, x, output_array, z, label_array, preds = trainers.calculate_predictions(model, train_loader)
    x_tilde2, x2, output_array2, z2, label_array2, preds2 = trainers.calculate_predictions(model, val_loader)

    # save clusters and images
    utils.sample_cluster(train_loader.dataset, preds, save_path, 10, 'Tissues_clusters_train')
    utils.sample_cluster(val_loader.dataset, preds2, save_path, 10, 'Tissues_clusters_eval')
    x_train = torch.cat((x, x_tilde), 2)
    x_eval = torch.cat((x2, x_tilde2), 2)

    save_image(x_train, os.path.join(save_path, 'x_train.png'))
    save_image(x_eval, os.path.join(save_path, 'x_eval.png'))
    
    # model metrics + saving
    nmi_train = utils.metrics.nmi(label_array, preds, average_method='arithmetic')
    ari_train = utils.metrics.ari(label_array, preds)
    acc_train = utils.metrics.acc(label_array, preds)
    nmi_eval = utils.metrics.nmi(label_array2, preds2, average_method='arithmetic')
    ari_eval = utils.metrics.ari(label_array2, preds2)
    acc_eval = utils.metrics.acc(label_array2, preds2)
    cm_train = cmx(label_array, preds)
    cm_eval = cmx(label_array2, preds2)

    np.savetxt(os.path.join(save_path, 'cm_train.csv'), cm_train, fmt='%10.5f')
    np.savetxt(os.path.join(save_path, 'cm_eval.csv'), cm_eval, fmt='%10.5f')
    data = {'NMI_train': nmi_train, 'ARI_train': ari_train, 'ACC_train': acc_train, 
            'NMI_eval': nmi_eval, 'ARI_eval': ari_eval, 'ACC_eval': acc_eval 
            }

    with open(os.path.join(save_path, 'metrics.csv'), 'w') as f:
        w = csv.DictWriter(f, data.keys())
        w.writeheader()
        w.writerow(data)

    pca = PCA(2, random_state=seed).fit(z)
    train_pca = pca.transform(z)
    eval_pca = pca.transform(z2)
    utils.PCA_plot(train_pca, label_array, preds, save_path, 'PCA_train')
    utils.PCA_plot(eval_pca, label_array2, preds2, save_path, 'PCA_eval')

    # t-sne
    t_train = TSNE(n_components = 2, random_state=seed).fit_transform(z)
    t_eval = TSNE(n_components = 2, random_state=seed).fit_transform(z2)
    utils.tsne_plot(t_train, label_array, preds, save_path, 'tsne_train')
    utils.tsne_plot(t_eval, label_array2, preds2, save_path, 'tsne_eval')

    

#-------------------------------------------------------------------------------------------------------------------------------------------------------

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

#-------------------------------------------------------------------------------------------------------------------------------------------------------

    # VaDE
    # method = 'VaDE'
    # labels = np.load(os.path.abspath(r"C:\Source\Research Repositories\TNBC\runs\VaDE_001\labels_e99.npy"))
    # preds = np.load(os.path.abspath(r"C:\Source\Research Repositories\TNBC\runs\VaDE_001\preds_e99.npy"))
    # z = np.load(os.path.abspath(r"C:\Source\Research Repositories\TNBC\runs\VaDE_001\z_e99.npy"))
    # n_clusters = 10
    # kmeans = KMeans(init = 'k-means++',n_clusters=n_clusters).fit(z)
    # kmeans.fit(z)
    # nmi_vae = normalized_mutual_info_score(labels, kmeans.labels_)
    # ari_vae = utils.metrics.ari(labels, kmeans.labels_)
    # acc_vae = utils.metrics.acc(labels, kmeans.labels_)
    # # plots
    # utils.PCA_plot(z, labels, kmeans.labels_, method, n_clusters, n_components = 2)
    # utils.tsne_plot(z, labels, kmeans.labels_, method, n_clusters, n_components = 2)

    #----------------------------------------------------------------------------------------------------------------------------------------------------
    
    pca = PCA(2, random_state=seed).fit(z)
    train_pca = pca.transform(z)
    test_pca = pca.transform(z2)
    km = KMeans(n_clusters, n_init=20).fit_predict(z)
    km2 = KMeans(n_clusters, n_init=20).fit_predict(z2)
    utils.PCA_plot(train_pca, label_array, preds, model_name, n_clusters)
    acc = utils.metrics.acc(label_array, preds)
    utils.naive_kmeans(train_pca, label_array, km, n_clusters) 
    # t-sne
    t_space = TSNE(n_components = 2, random_state=seed).fit_transform(z)
    utils.tsne_plot(t_space, label_array, preds, model_name, n_clusters)

    # fig, axs = plt.subplots(2,2)
    # axs = axs.ravel()
    # # color = cm.jet(np.linspace(0, np.max(output_array), num=output_array.shape[0]))
    # for idx in range(output_array.shape[1]):
    #     # color = cm.jet(output_array[:, idx])#np.linspace(0, np.max(output_array[:, idx]), num=output_array.shape[0]))
    #     color = cm.jet(np.linspace(0, np.max(output_array[:,idx]), num=output_array.shape[0]))
    #     test = axs[idx].scatter(t_space[:, 0], t_space[:, 1], c=color)
    #     axs[idx].set_title('Heatmap cluster label ' + str(idx))
    #     fig.colorbar(test, ax=axs[idx])
    # plt.show()

    # for idx in range(output_array.shape[1]):
    #     sns.kdeplot(output_array[:,idx])
    # plt.xlabel('x')
    # plt.ylabel('prob. size')
    # plt.title('initial p_ij')
    # plt.show()

    
    # # simple CAE for TNBC
    # model_name = 'TNBC_CAE'
    # img_size = [3, 512, 512]
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # to_eval = "mods." + model_name + "(img_size)"
    # model = eval(to_eval)
    # model = model.to(device)
    # model.load_state_dict(torch.load(r"C:\Source\Research Repositories\TNBC\nets\TNBC_CAE_001.pt"))
    # model.to(device)
    # val_batch_size = 4

    # transform = transforms.Compose([transforms.ToTensor()])
    # val = datagen.DataGen_TNBC('eval', r"Z:\Speciale\sampling_eval", transform)
    # val_loader = data.DataLoader(val, val_batch_size, shuffle=False)
    # loss_function = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    # losses = []
    # model.eval()
    # z1 = []
    # labels = []
    # progress = tqdm(enumerate(val_loader), desc="Loss: ", total=len(val_loader))
    # for i, data in progress:
    #     X, y = data[0].to(device), data[1]
    #     model.zero_grad()
    #     emb = model.embed(X)
    #     x_tilde = model(X)
    #     labels.append(y)
    #     z1.append(emb.detach().cpu().numpy())
    # z1 = np.concatenate(z1)
    # tsne = TSNE(n_components = 2, random_state=seed)
    # t_space = tsne.fit_transform(z1[:,:,0,0])
    # # utils.findK(t_space, 20)    #Determine best # of clusters in data
    # linked = linkage(t_space, 'single')

    # labelList = range(1, z1.shape[0]-100)

    # plt.figure(figsize=(10, 7))
    # dendrogram(linked,
    #             truncate_mode='lastp',
    #             orientation='top',
    #             labels=labelList,
    #             distance_sort='descending',
    #             show_leaf_counts=True)
    # plt.title('Hierarchical Clustering Dendrogram (truncated)')
    # plt.xlabel('sample index or (cluster size)')
    # plt.ylabel('distance')
    # plt.show()
    # utils.dendrofile(t_space[:,:,0,0])

    
    
    # save_image(x_tilde, os.path.join(r"C:\Source\Research Repositories\TNBC\runs\TNBC_CAE_001\x_tilde.png"))
    # save_image(X, os.path.join(r"C:\Source\Research Repositories\TNBC\runs\TNBC_CAE_001\x.png"))
    # utils.TNBC_PCA(z1[:,:,0,0], labels, model_name)
    # utils.TNBC_tsne(z1[:,:,0,0], labels, model_name)



    


