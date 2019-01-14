import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from IPython.display import display


def readData():

    data = pd.read_csv('data/sm13-14.csv', error_bad_lines=False, sep=';', na_values=['0'],  usecols=['SendingCountry', 'MobilityType', 'SubjectAreaCode', 'DurationInMonths', 'LevelOfStudy', 'ParticipantGender', 'Language', 'SendingPartnerErasmusID', 'HostingPartnerCountry'])
    #data = pd.read_csv('data/sm13-14.csv', error_bad_lines=False, sep=';', na_values=['0'])
    return data

def selectData(data, column, value):
    return data.loc[data[column] == value]

def checkDataValues(data):
    for col in data.columns.values:
        print(col, data[col].unique())

def printMissingValues(dataset):
    print("\nBrakujące dane w poszczególnych kolumnach:")
    print("------------------------------------------")
    print("Nazwa kolumny            Liczba braków\n")
    print(dataset.isnull().sum())
    print("------------------------------------------")

def deleteRowsWithMissingValues(dataset):
    print("\nKasowanie rekordów zawierających braki...")
    print("Liczba rekordów w bazie przed kasowaniem: %d" % dataset.shape[0])
    changedDataset = dataset.dropna(axis=0, how='any')
    print("Liczba rekordów w bazie po kasowaniu: %d" % changedDataset.shape[0])
    print("Usunieto %d rekordów" % (dataset.shape[0] -changedDataset.shape[0]))
    return changedDataset

def encodeData(data):    
    columns = data.columns.values
    for col in columns:
        le = LabelEncoder()
        data[col+'_code'] = le.fit_transform(data[col])
    #oe = OrdinalEncoder()
    #data = oe.fit_transform(data.values.tolist())
    return data

def writeDataFrameToCSV(data, filename):
    data.to_csv('data/'+filename, sep=';', encoding='utf-8', index=False)

def getCodesFromData(data):
    display(data.describe())
    dataCodes = data[['SendingCountry_code', 'MobilityType_code', 'SubjectAreaCode_code', 'DurationInMonths_code', 'LevelOfStudy_code', 'ParticipantGender_code', 'Language_code', 'SendingPartnerErasmusID_code', 'HostingPartnerCountry_code']].copy()
    return dataCodes

data = readData()
print(data.dtypes)
#data = selectData(data, 'HostingPartnerCountry', 'PL')
checkDataValues(data)
printMissingValues(data)
data = deleteRowsWithMissingValues(data)
normalizedData = encodeData(data)
codesData = getCodesFromData(normalizedData)
writeDataFrameToCSV(codesData, 'normalized.csv')
