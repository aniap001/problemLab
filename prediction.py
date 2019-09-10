import pickle
import pandas as pd
from sklearn import model_selection  
import numpy as np


# load the model from disk
filename = 'Decision Tree' + ' model'

def readData():
    data = pd.read_csv('data/do_klasyfikacji/label_encoded_no_reduction.csv', error_bad_lines=False, sep=';')
    return data


def logic():
    data = readData()
    x = data.drop('ReceivingCountry', axis=1)
                
    y = data['ReceivingCountry']
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.8)
    loaded_model = pickle.load(open('model/' + filename, 'rb'))
    result = loaded_model.score(x_test, y_test)
    print(result)

    prediction = loaded_model.predict(X=x_test)
    correct = prediction[prediction == np.array(y_test)]
    percentage = 100 * len(correct) / len(prediction)
    print("Procent poprawnie sklasyfikowanych rekordów: {} %".format(percentage))

logic()