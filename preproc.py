import pandas as pd
import numpy as np

from IPython.display import display
        

def readData():
    data = pd.read_csv('data/sm13-14.csv', error_bad_lines=False, na_values=['Not known or unspecified'], decimal=',', sep=';', usecols=['SendingCountry', 'MobilityType', 'SpecialNeeds', 'SubjectAreaName', 'CombinedMobilityYesNo', 'StartDate', 'EndDate', 'DurationInMonths', 'DurationInDays', 'SubsistenceTravel', 'LevelOfStudy', 'ParticipantType', 'ParticipantGender', 'Language', 'SendingPartnerErasmusID', 'HostingPartnerErasmusID', 'HostingPartnerCity', 'ReceivingCountry'])
    #data = pd.read_csv('data/sm13-14.csv', error_bad_lines=False, sep=';', na_values=['0'],  usecols=['SendingCountry', 'MobilityType', 'SubjectAreaCode', 'DurationInMonths', 'LevelOfStudy', 'ParticipantGender', 'Language', 'SendingPartnerErasmusID', 'HostingPartnerCountry'])
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
    dataset.isnull().sum().to_frame().to_csv('data/generated_data.csv')
    print("------------------------------------------")

def deleteRowsWithMissingValues(dataset):
    print("\nKasowanie rekordów zawierających braki...")
    print("Liczba rekordów w bazie przed kasowaniem: %d" % dataset.shape[0])
    changedDataset = dataset.dropna(axis=0, how='any')
    print("Liczba rekordów w bazie po kasowaniu: %d" % changedDataset.shape[0])
    print("Usunieto %d rekordów" % (dataset.shape[0] -changedDataset.shape[0]))
    return changedDataset

def deleteColumns(data, columns):
    return data.drop(columns=columns)

def clearData(data):
    print('Typy danych:')
    print(data.dtypes)
    print('--------------------------------')
    printMissingValues(data)
    data = deleteColumns(data, ['CombinedMobilityYesNo', 'HostingPartnerErasmusID'])
    data = deleteRowsWithMissingValues(data)
    return data

def writeDataFrameToCSV(data, filename):
    data.to_csv('data/'+filename, sep=';', encoding='utf-8', index=False)


def exploreColumns(data):
    print(data['SpecialNeeds'].describe())
    print(data['SubsistenceTravel'].describe())

def howManyUnique(data):
    for col in data.columns.values:
        print (str(col) + ': ' + str(len(data[col].unique().tolist())))

def transformToBinary(data):
    data.SpecialNeeds[data.SpecialNeeds == 0] = 0
    data.SpecialNeeds[data.SpecialNeeds != 0] = 1
    data = data.astype({'SpecialNeeds' : int})
    return data
def logic():

    data = readData()
    #data.dtypes.to_csv('../test.csv')
    
    #####data = selectData(data, 'HostingPartnerCountry', 'PL')
    data = clearData(data)
    exploreColumns(data)
    howManyUnique(data)
    print(data.dtypes)
    checkDataValues(data)
    data = deleteColumns(data, {'StartDate', 'EndDate', 'DurationInDays', 'ParticipantType'})

    data = transformToBinary(data)
    writeDataFrameToCSV(data, 'processed.csv')

logic()