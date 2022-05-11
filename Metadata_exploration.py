# Exploration of TCGA metadata for triple negative breast cancer patients
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from random import seed
import matplotlib.pyplot as plt
from collections import Counter
from lifelines import KaplanMeierFitter

# path = os.path.abspath(r"C:\Source\Research Repositories\TNBC\data\GDC_metadata_exclusive.xlsx")
# df = pd.read_excel(path, header=[1,])
# pam50 = Counter(df.PAM50)
# pam50_lite = Counter(df.PAM50lite)
# TNBC_4 = Counter(df.TNBCtype_4)

#---------------------------------------------------------------------------------------------------------
# # pie chart for PAM50 and TNBC
# plt.subplot(1,2,1)
# pam_sizes = pam50.values()
# pam_labels = pam50.keys()
# colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'white']
# plt.pie(pam_sizes, labels=pam_labels, colors=colors,
# autopct='%1.1f%%', shadow=True, startangle=140, textprops={'fontsize': 18})
# plt.axis('equal')
# plt.title('PAM50', fontsize=25)
# plt.subplot(1,2,2)
# TNBC_sizes = TNBC_4.values()
# TNBC_labels = TNBC_4.keys()
# colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'white']
# plt.pie(TNBC_sizes, labels=TNBC_labels, colors=colors,
# autopct='%1.1f%%', shadow=True, startangle=140, textprops={'fontsize': 18})
# plt.axis('equal')
# plt.title('TNBC type 4', fontsize=25)
# plt.show()

#------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
# # pie chart for PAM50_lite and TNBC
# plt.subplot(1,2,1)
# pam_sizes = pam50_lite.values()
# pam_labels = pam50_lite.keys()
# colors =  ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'white']
# plt.pie(pam_sizes, labels=pam_labels, colors=colors, radius=1,
# autopct='%1.1f%%', shadow=True, startangle=140, textprops={'fontsize': 18})
# plt.axis('equal')
# plt.title('PAM50, n=106', fontsize=25)

# plt.subplot(1,2,2)
# TNBC_sizes = TNBC_4.values()
# TNBC_labels = TNBC_4.keys()
# colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'white']
# plt.pie(TNBC_sizes, labels=TNBC_labels, colors=colors, radius=1,
# autopct='%1.1f%%', shadow=True, startangle=140, textprops={'fontsize': 18})
# plt.axis('equal')
# plt.title('TNBC Lehman type 4, n=106', fontsize=25)
# plt.show()

#---------------------------------------------------------------------------------------------------------------

# # KM survival curves
# deaths = df.vitalstatus_TNBC
# time_death = df.yearstodeath_TNBC
# kmf = KaplanMeierFitter() 
# ## Fit the data into the model
# kmf.fit(time_death, deaths)
# ## Create an estimate
# kmf.plot(ci_show=False, linewidth=3)
# plt.legend(['KM curve'], fontsize=15)
# plt.xlabel('Time (years)', fontsize=15)
# plt.ylabel('Fraction survival', fontsize=15)
# plt.title('Kaplan-Meier curve TNBC', fontsize=25)
# plt.show()


# Pam 50 KM curves
# for i, subs in enumerate(pam50.keys()):
#     ax = plt.subplot(1,1,1)
#     idx = np.where(df.PAM50==subs)
#     # kmf = KaplanMeierFitter()
#     kmf.fit(time_death[idx[0]], deaths[idx[0]], label=subs)
#     kmf.plot(ax=ax, ci_show=False, legend=False)
#     plt.title('Kaplan-Meier survival curves PAM50')
#     plt.xlim(0, 10)
#     plt.legend(pam50.keys(), loc=3)
#     plt.ylabel('Fraction survival')
#     plt.xlabel('Time (years)')
# plt.show()

# TNBC_4 type KM curves
# for i, subs in enumerate(TNBC_4.keys()):
#     ax = plt.subplot(1,1,1)
#     idx = np.where(df.TNBCtype_4==subs)
#     # kmf = KaplanMeierFitter()
#     kmf.fit(time_death[idx[0]], deaths[idx[0]], label=subs)
#     kmf.plot(ax=ax, ci_show=False, legend=False)
#     plt.title('Kaplan-Meier survival curves Lehman 4 class')
#     plt.xlim(0, 10)
#     plt.legend(pam50.keys(), loc=3)
#     plt.ylabel('Fraction survival')
#     plt.xlabel('Time (years)')
# plt.show()

# ------------------------------------------------------------------------------------------------------------
# plotting internals for MNIST, gamma=0.1
# nmi_path = os.path.abspath(r"Z:\Speciale\Results\Results\Vary_gamma\MNIST\CAE_3_gamma_0.1\gamma_0.1-tag-NMI.csv")
# ari_path = os.path.abspath(r"Z:\Speciale\Results\Results\Vary_gamma\MNIST\CAE_3_gamma_0.1\gamma_0.1-tag-ARI.csv")
# acc_path = os.path.abspath(r"Z:\Speciale\Results\Results\Vary_gamma\MNIST\CAE_3_gamma_0.1\gamma_0.1-tag-Acc.csv")
# save_path = os.path.abspath(r"Z:\Speciale\Results\Results\Vary_gamma\MNIST\CAE_3_gamma_0.1")
# nmi_data = pd.read_csv(nmi_path)
# ari_data = pd.read_csv(ari_path)
# acc_data = pd.read_csv(acc_path)
# plt.subplots(figsize=(10, 6))
# plt.plot(nmi_data.Step, nmi_data.Value, linewidth=3)
# plt.plot(ari_data.Step, ari_data.Value, linewidth=3)
# plt.plot(acc_data.Step, acc_data.Value, linewidth=3)
# plt.xlabel('T steps', fontsize=15)
# plt.ylabel('Score', fontsize=15)
# plt.legend(('NMI', 'ARI', 'ACC'))
# plt.title('Internal metrics for ' + r'$\gamma=0.1$', fontsize=20)
# plt.savefig(os.path.join(save_path, 'Internals_vs_T.png'))

# plotting externals for MNIST
# nmi_path = os.path.abspath(r"Z:\Speciale\Results\Results\Vary_gamma\MNIST\CAE_3_gamma_1\run-MNIST_CAE_3_gamma_1-tag-NMI.csv")
# ari_path = os.path.abspath(r"Z:\Speciale\Results\Results\Vary_gamma\MNIST\CAE_3_gamma_1\run-MNIST_CAE_3_gamma_1-tag-ARI.csv")
# acc_path = os.path.abspath(r"Z:\Speciale\Results\Results\Vary_gamma\MNIST\CAE_3_gamma_1\run-MNIST_CAE_3_gamma_1-tag-ACC.csv")
# save_path = os.path.abspath(r"Z:\Speciale\Results\Results\Vary_gamma\MNIST\CAE_3_gamma_1")
# nmi_data = pd.read_csv(nmi_path)
# ari_data = pd.read_csv(ari_path)
# acc_data = pd.read_csv(acc_path)
# plt.subplots(figsize=(10, 6))
# plt.plot(nmi_data.Step, nmi_data.Value, linewidth=3)
# plt.plot(ari_data.Step, ari_data.Value, linewidth=3)
# plt.plot(acc_data.Step, acc_data.Value, linewidth=3)
# plt.xlabel('T steps', fontsize=15)
# plt.ylabel('Score', fontsize=15)
# plt.legend(('NMI', 'ARI', 'ACC'))
# plt.title('External metrics, MNIST, ' + r'$\gamma=1$', fontsize=20)
# plt.savefig(os.path.join(save_path, 'MNIST_Externals_vs_T.png'))

# plotting externals for Tissues
nmi_path = os.path.abspath(r"Z:\Speciale\Results\Results\Vary_gamma\Tissues\Tissues_TNBC_CAE_3_gamma_1\gamma_1-tag-NMI.csv")
ari_path = os.path.abspath(r"Z:\Speciale\Results\Results\Vary_gamma\Tissues\Tissues_TNBC_CAE_3_gamma_1\gamma_1-tag-ARI.csv")
acc_path = os.path.abspath(r"Z:\Speciale\Results\Results\Vary_gamma\Tissues\Tissues_TNBC_CAE_3_gamma_1\gamma_1-tag-ACC.csv")
save_path = os.path.abspath(r"Z:\Speciale\Results\Results\Vary_gamma\Tissues\Tissues_TNBC_CAE_3_gamma_1")
nmi_data = pd.read_csv(nmi_path)
ari_data = pd.read_csv(ari_path)
acc_data = pd.read_csv(acc_path)
plt.subplots(figsize=(10, 6))
plt.plot(nmi_data.Step, nmi_data.Value, linewidth=3)
plt.plot(ari_data.Step, ari_data.Value, linewidth=3)
plt.plot(acc_data.Step, acc_data.Value, linewidth=3)
plt.xlabel('T steps', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.legend(('NMI', 'ARI', 'ACC'))
plt.title('External metrics, Tissues, ' + r'$\gamma=1$', fontsize=20)
plt.savefig(os.path.join(save_path, 'Tiss_Externals_vs_T.png'))

hej = 3