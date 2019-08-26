# Exploration of TCGA metadata for triple negative breast cancer patients
import os
import numpy as np
import pandas as pd
path = os.path.abspath(r'C:\\Source\\Research Repositories\\TNBC\Data\\GDC patients with triple negative breast cancer.xlsx')
# path = os.path.abspath('.TNBC\\Data\\GDC patients with triple negative breast cancer.xlsx')
df = pd.read_excel(path, header=[1,])
hej1 = df.loc[df['CPE'] > 0.6]
hej2 = df.loc[(df['CPE'] < 0.6) & (df['CHAT.purity'] >= 0.6)]
new_df = df.loc[df['CPE'] > 0.6].append(df.loc[(df['CPE'] < 0.6) & (df['CHAT.purity'] >= 0.6)])
hej = 2
# IHC col 19 and CHAT col 21
# patients indexes we don't have slides from:
#row_del = [23, 52, 63, 65, 72, 80, 84, 97, 105, 113, 117, 123, 129, 130, 157, 179]

