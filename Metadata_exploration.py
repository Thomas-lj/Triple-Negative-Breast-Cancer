# Exploration of TCGA metadata for triple negative breast cancer patients
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from random import seed

# path = os.path.abspath(r'C:\\Source\\Research Repositories\\TNBC\Data\\GDC patients with triple negative breast cancer.xlsx')
# # path = os.path.abspath('.TNBC\\Data\\GDC patients with triple negative breast cancer.xlsx')
# df = pd.read_excel(path, header=[1,])
# hej1 = df.loc[df['CPE'] > 0.6]
# hej2 = df.loc[(df['CPE'] < 0.6) & (df['CHAT.purity'] >= 0.6)]
# new_df = df.loc[df['CPE'] > 0.6].append(df.loc[(df['CPE'] < 0.6) & (df['CHAT.purity'] >= 0.6)])
# IHC col 19 and CHAT col 21
# patients indexes we don't have slides from:
#row_del = [23, 52, 63, 65, 72, 80, 84, 97, 105, 113, 117, 123, 129, 130, 157, 179]

# making .csv file from data
dat_path = os.path.abspath(r'Z:\Speciale\TNBC\Data')
csv_path = os.path.abspath(r'C:\Source\Research Repositories\TNBC\data\dataframe_512.csv')

folders = os.listdir(dat_path)
IDs = []
paths = []
for folder in folders:
    ID = os.listdir(os.path.join(dat_path, folder))
    for idx in enumerate(ID):
        path = os.path.join(dat_path, folder, idx[1])
        paths.append(path)
    IDs.append(ID)
IDs = np.concatenate(IDs)
df = pd.DataFrame(columns= ['File', 'FileID', 'Group']).fillna(0)
df.File = paths
df.FileID = IDs

split = [0.9, 0.10]
unique_subjects = sorted(list(set(df['File'])))
seed(1234)
trainID, evalID = train_test_split(unique_subjects, test_size = split[1])
for id in df['File']:
    if id in trainID:
        df.loc[df['File'] == id, 'Group'] = 'train'
    elif id in evalID:
        df.loc[df['File'] == id, 'Group'] = 'eval'
    else:
        print('No subset assignment for {}.'.format(id))
pd.DataFrame(df).to_csv(csv_path)
