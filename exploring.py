import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import display

def readData():

    data = pd.read_csv('data/normalized.csv', error_bad_lines=False, sep=';')
    return data

def writeDataFrameToCSV(data, filename):
    data.to_csv(filename, sep=';')

def correlationMatrix(data):
    correlation_matrix = data.corr(method='spearman')
    plt.figure(figsize=(8,8))
    ax = sns.heatmap(correlation_matrix, vmax=1, square=True, annot=True,fmt='.2f', cmap ='GnBu', cbar_kws={"shrink": .5}, robust=True)
    plt.title('Macierz korelacji pomiędzy cechami', fontsize=20)
    plt.show()

def display_corr_with_col(df, col):
    correlation_matrix = df.corr(method='spearman')
    correlation_type = correlation_matrix[col].copy()
    abs_correlation_type = correlation_type.apply(lambda x: abs(x))
    desc_corr_values = abs_correlation_type.sort_values(ascending=False)
    y_values = list(desc_corr_values.values)[1:]
    x_values = range(0,len(y_values))
    xlabels = list(desc_corr_values.keys())[1:]
    fig, ax = plt.subplots(figsize=(8,8))
    ax.bar(x_values, y_values)
    ax.set_title('Korelacje wszystkich cech z {}'.format(col), fontsize=15)
    ax.set_ylabel('Współczynnik korelacji Spearmana (wartość bezwzględna)', fontsize=13)
    plt.xticks(x_values, xlabels, rotation='vertical')
    plt.show()

def showDataVariance(data):
    columns = list(data.columns.values)
    X = data[columns].values
    X_std = StandardScaler().fit_transform(X)
    
    pca = PCA().fit(X_std)
    var_ratio = pca.explained_variance_ratio_
    components = pca.components_
    #print(pca.explained_variance_)
    plt.plot(np.cumsum(var_ratio))
    plt.xlim(0,9,1)
    plt.xlabel('Liczba cech', fontsize=16)
    plt.ylabel('Cumulative explained variance', fontsize=16)
    plt.show()

data = readData()

correlationMatrix(data)
showDataVariance(data)
display_corr_with_col(data, 'HostingPartnerCountry_code')