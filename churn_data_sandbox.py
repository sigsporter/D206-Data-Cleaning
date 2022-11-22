import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as pp
import seaborn as sns
from sklearn.decomposition import PCA
df = pd.read_csv(
    'C:/Users/sigsp/OneDrive/Desktop/D206 Data Cleaning/churn_raw_data.csv')

# Prepping columns: Dropping Unnamed: 0 column, setting index, renaming item1
# to item8 columns
df.drop(columns='Unnamed: 0', inplace=True)
df.set_index('CaseOrder', inplace=True)
survey_dict = {'item1': 'Response',
               'item2': 'Fix',
               'item3': 'Replacement',
               'item4': 'Reliability',
               'item5': 'Options',
               'item6': 'Respectful',
               'item7': 'Courteous',
               'item8': 'Listening'}

df.rename(columns=survey_dict, inplace=True)

numericCols = df.select_dtypes(exclude=['object']).columns.tolist()

dfNumeric = df[numericCols]

numericCols_normalized = (dfNumeric - dfNumeric.mean())/dfNumeric.std()

pca = PCA(n_components=dfNumeric.shape[1])

PCcols = []
i = 1

while i < len(numericCols)+1:
   col = 'PC'+str(i)
   PCcols.append(col)
   i+=1
print(PCcols)
# pca.fit(numericCols_normalized)
# numeric_pca = pd.DataFrame(pca.transform(numericCols_normalized), columns=numericCols)

# cov_matrix = np.dot(numericCols_normalized.T, numericCols_normalized)/dfNumeric.shape[0]
# eigenvalues = [np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)) for eigenvector in pca.components_]
# pp.plot(eigenvalues)
# pp.xlabel('number of components')
# pp.ylabel('eigenvalue')
# pp.show()

# loading = pd.DataFrame(pca.components_.T, columns=numericCols, index=dfNumeric.columns)
# print(loading)
