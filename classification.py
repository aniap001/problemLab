import pandas as pd
import numpy as np
from time import process_time
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection  
from sklearn import metrics


import pickle

#from IPython.display import display

dict_classifiers = {
    #"Decision Tree": DecisionTreeClassifier(max_depth=30, max_features=180),
    "Neural Network": MLPClassifier(activation='tanh', hidden_layer_sizes={40, 50}, solver='adam', verbose=True, max_iter=150),
    #"Naive Bayes": GaussianNB(),
    #"Nearest Neighbors": KNeighborsClassifier(),
    #"Linear SVM": SVC(),
}

import warnings
#warnings.filterwarnings("ignore")

def readData():
    data = pd.read_csv('data/do_klasyfikacji/label_encoded_fs_reduction.csv', error_bad_lines=False, sep=';')
    return data

def checkDataValues(data):
    for col in data.columns.values:
        print(col, data[col].unique())

#funkcja redukująca ilość rekordów w bazie - dla szybszych obliczeń
def getLimitedData(df, ratio):
    mask = np.random.rand(len(df)) < ratio 
    limited_df = df[mask]
    return limited_df
 

def classification(data, classifier, classifier_name, class_col):
    #print(data.loc[:,'LevelOfStudy':].head())
    #dzielimy zbior danych na dane treningowe i testowe w proporcji 8:2
    x = data.drop(class_col, axis=1)
        
    y = data[class_col]
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.8, shuffle=True)

    
    #fitting
    clf = classifier
    #clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    print(clf)
    t_start = process_time()
    clf.fit(x_train, y_train)
    t_end = process_time()   
    fit_diff = t_end - t_start

    #prediction
    t_start = process_time()
    prediction = clf.predict(X=x_test)
    t_end = process_time()  
    pred_diff = t_end - t_start

    t_start = process_time()
    prediction_trainset = clf.predict(X=x_train)
    t_end = process_time()  
    pred_diff = t_end - t_start


    #accuracy
    correct = prediction[prediction == np.array(y_test)]
    percentage = 100 * len(correct) / len(prediction)

    correct_trainset = prediction_trainset[prediction_trainset == np.array(y_train)]
    percentage_trainset = 100 * len(correct_trainset) / len(prediction_trainset)
    print("Procent poprawnie sklasyfikowanych rekordów: {} %".format(percentage))
    print("Procent poprawnie sklasyfikowanych rekordów zbioru treningowego: {} %".format(percentage_trainset))
    print("Czas trenowania klasyfikatora: {} s".format(fit_diff))
    print("Czas predykcji klas: {} s".format(pred_diff))

    if classifier_name == "Decision Tree":
        print("Głębokość drzewa: {}".format(clf.tree_.max_depth))



    labels = ['NO' 'IE' 'ES' 'SE' 'DE' 'NL' 'PL' 'TR' 'BE' 'DK' 'IT' 'FI' 'CZ' 'PT'
    'CH' 'SI' 'GB' 'FR' 'HR' 'BG' 'GR' 'LT' 'MT' 'IS' 'LV' 'EE' 'CY' 'RO'
    'HU' 'SK' 'LI' 'LU' 'AT' 'MK']
    pretty_cm(prediction, y_test, labels)
    return clf


def pretty_cm(y_pred, y_truth, labels):
    # pretty implementation of a confusion matrix
    cm = metrics.confusion_matrix(y_truth, y_pred)
    ax= plt.subplot()
    sns.heatmap(cm, annot=False, fmt="d", linewidths=.5, square = True, cmap = 'BuGn_r')
    # labels, title and ticks
    ax.set_xlabel('Przewidziana klasa')
    ax.set_ylabel('Rzeczywista klasa')
    ax.set_title('Skuteczność: {0}'.format(metrics.accuracy_score(y_truth, y_pred)), size = 22) 
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    plt.show()

def writeDataFrameToCSV(data, filename):
    data.to_csv('data/'+filename, sep=';', encoding='utf-8', index=False)

def takeLastNRecordsForlassification(data, n):
    print(data.shape)
    forClassification = data.tail(n)
    data = data.iloc[:-n]
    print(data.shape)
    writeDataFrameToCSV(forClassification, 'do_predykcji/last_3.csv')
    return data

def logic():

    data = readData()
    #print(data.head())
    #data = getLimitedData(data, 0.5)
    #data = data.drop(['Language'], axis=1)
    data = takeLastNRecordsForlassification(data, 10)

    for classifier_name, classifier in list(dict_classifiers.items()):
        print('Budowanie modelu klasyfikatora ' + classifier_name)
        clf = classification(data, classifier, classifier_name, 'ReceivingCountry')
        filename = classifier_name + ' model'
        pickle.dump(clf, open('model/' + filename + '.sav', 'wb'))
        


    #test(tree_clf, [['27','1','64','2','0','0','14','1827','1299','587']])
    #checkDataValues(data)
    #classification(data)

logic()
