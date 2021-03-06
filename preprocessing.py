import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
from dython import nominal
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from time import process_time
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def readData():
    data = pd.read_csv('data/processed.csv', error_bad_lines=False, sep=';')
    return data

def writeDataFrameToCSV(data, filename):
    data.to_csv(filename, sep=';')

def correlationMatrix(data, encoded):
    
    if (encoded):
        correlation_matrix = data.corr(method='spearman')
        plt.figure(figsize=(8,8))
        ax = sns.heatmap(correlation_matrix, vmax=1, square=True, annot=True, fmt='.2f', cmap ='GnBu', cbar_kws={"shrink": .5}, robust=True)
        plt.title('Macierz korelacji pomiędzy cechami', fontsize=20)
        plt.show()
    else:
        nominal.associations(data, theil_u=True, nominal_columns=['SendingCountry', 'ReceivingCountry', 'MobilityType', 'SpecialNeeds', 'SubjectAreaName', 'LevelOfStudy', 'ParticipantGender', 'Language', 'SendingPartnerErasmusID', 'HostingPartnerCity'])


def showDataVariance(data):
    columns = list(data.columns.values)
    X = data[columns].values
    X_std = StandardScaler().fit_transform(data)
    
    pca = PCA().fit(X_std)
    var_ratio = pca.explained_variance_ratio_
    components = pca.components_
    #print(pca.explained_variance_)
    plt.plot(np.cumsum(var_ratio))
    #plt.xlim(0,8,1)
    plt.xlabel('Liczba cech', fontsize=16)
    plt.ylabel('Wyjaśniona wariancja', fontsize=16)
    plt.show()

def deleteColumns(data, columns):
    return data.drop(columns=columns)

def howManyUnique(data):
    for col in data.columns.values:
        print (str(col) + ': ' + str(len(data[col].unique().tolist())))

def checkDataValues(data):
    for col in data.columns.values:
        print(col, data[col].unique())

def encodeData(data, method, columns):
    encodedData = data.copy()
    if method == 'one-hot-encoding':
        encodedData = pd.get_dummies(data, columns=columns)
        print(encodedData.head())
        print(encodedData.shape)
    elif method == 'label-encoding':
        le = LabelEncoder()
        for col in columns:
            encodedData[col] = le.fit_transform(data[col])

    return encodedData



def normalizeColumns(data, columns):
    data = data.astype(float)
    t_start = process_time()
    for col in columns:
        column = data[col].copy()
        min = column.min()
        max = column.max()
        normMin = 0
        normMax = 1
        for i in range(column.size):
            column[i] = normMin + (column[i] - min)*(normMax - normMin)/(max-min)
        data[col] = column
    t_end = process_time()
    print('\nNormalizacja danych trwała {} s \n'.format(t_end-t_start))

    return data

def writeDataFrameToCSV(data, filename):
    t_start = process_time()
    data.to_csv('data/'+filename, sep=';', encoding='utf-8', index=False)
    t_end = process_time()
    print('\nZapisywanie danych trwało {} s \n'.format(t_end-t_start))

def transformToBinary(data):
    data.SpecialNeeds[data.SpecialNeeds == 0] = 0
    data.SpecialNeeds[data.SpecialNeeds != 0] = 1
    data = data.astype({'SpecialNeeds' : int})
    return data

def pca_reduction(data, n_components):
    pca = PCA(n_components=n_components).fit(data)
    data = pca.transform(data)
    data = pd.DataFrame(data)
    return data

def logic():

    data = readData()

    data = deleteColumns(data, ['SendingPartnerErasmusID', 'HostingPartnerCity'])

    data = transformToBinary(data)

    class_column = data['ReceivingCountry']
    data = deleteColumns(data, ['ReceivingCountry'])
    data = encodeData(data, 'label-encoding', ['MobilityType', 'SubjectAreaName', 'DurationInMonths', 'LevelOfStudy', 'ParticipantGender', 'Language', 'SendingCountry'])
    
    correlationMatrix(data, True)
    
    data = normalizeColumns(data, ['SubsistenceTravel', 'MobilityType', 'SubjectAreaName', 'DurationInMonths', 'LevelOfStudy', 'ParticipantGender', 'Language', 'SendingPartnerErasmusID'])
    #data = normalizeColumns(data, ['SubsistenceTravel'])
    showDataVariance(data)
    data = pca_reduction(data, 200)

    #data = deleteColumns(data, ['MobilityType', 'SpecialNeeds', 'LevelOfStudy', 'ParticipantGender'])

    data['ReceivingCountry'] = class_column
    writeDataFrameToCSV(data, 'do_klasyfikacji/one_hot_encoded_no_reduction.csv')
    
    #display_corr_with_col(data, 'ReceivingCountry')

logic()
