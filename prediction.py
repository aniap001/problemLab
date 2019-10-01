import pickle
import pandas as pd
from sklearn import model_selection  
import numpy as np


# load the model from disk
filename = 'Neural Network' + ' model.sav'

def readData():
    data = pd.read_csv('data/do_predykcji/last_3.csv', error_bad_lines=False, sep=';')
    return data


def logic():
    data = readData()
    print(data)
    y = data['ReceivingCountry']
    x = data.drop('ReceivingCountry', axis=1)
                
    
    #x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.8)
    loaded_model = pickle.load(open('model/' + filename, 'rb'))
    #result = loaded_model.score(x, y)

    prediction = loaded_model.predict(X=x)
    print("\nRzeczywiste klasy")
    print(np.array(y))
    print ("\nPrzewidziano nastepujace klasy:")
    print(prediction)

    correct = prediction[prediction == np.array(y)]
    percentage = 100 * len(correct) / len(prediction)
    print("\nProcent poprawnie sklasyfikowanych rekordów: {} %".format(percentage))

logic()