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
    "Decision Tree": DecisionTreeClassifier(max_depth=15, max_features=5),
    #"Neural Network": MLPClassifier(activation='tanh', hidden_layer_sizes={50, 100}, solver='sgd', verbose=True, max_iter=150),
    #"Naive Bayes": GaussianNB(),
    #"Nearest Neighbors": KNeighborsClassifier(),
    #"Linear SVM": SVC(),
}

import warnings
#warnings.filterwarnings("ignore")

def readData():
    data = pd.read_csv('data/do_klasyfikacji/label_encoded_pca_reduction.csv', error_bad_lines=False, sep=';')
    return data

def checkDataValues(data):
    for col in data.columns.values:
        print(col, data[col].unique())

#funkcja redukująca ilość rekordów w bazie - dla szybszych obliczeń
def getLimitedData(df, ratio):
    mask = np.random.rand(len(df)) < ratio 
    limited_df = df[mask]
    return limited_df

    


# #podziel na zbiory treningowe i testowe
# def get_train_test(df, y_col, x_cols, ratio):
#     mask = np.random.rand(len(df)) < ratio
#     df_train = df[mask]
#     df_test = df[~mask]
       
#     Y_train = df_train[y_col].values
#     Y_test = df_test[y_col].values
#     X_train = df_train[x_cols].values
#     X_test = df_test[x_cols].values
#     #return df_train, df_test, X_train, Y_train, X_test, Y_test
#     return X_train, Y_train, X_test, Y_test
 

def classification(data, classifier, classifier_name, class_col):
    #print(data.loc[:,'LevelOfStudy':].head())
    #dzielimy zbior danych na dane treningowe i testowe w proporcji 8:2
    x = data.drop(class_col, axis=1)
        
    y = data[class_col]
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.8, shuffle=True)

    # parameter_space = {
    #     'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    #     'activation': ['tanh', 'relu'],
    #     'solver': ['sgd', 'adam'],
    #     'alpha': [0.0001, 0.05],
    #     'learning_rate': ['constant','adaptive'],
    # }

    
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

    # Best paramete set
    #print('Best parameters found:\n', clf.best_params_)

    # All results
    #means = clf.cv_results_['mean_test_score']
    #stds = clf.cv_results_['std_test_score']
    #for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

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
    #pretty_cm(prediction, y_test, labels)
    return clf

# def classification_with_cross_validation(data, classifier, class_col):
#     #dzielimy zbior danych na dane treningowe i testowe w proporcji 8:2
#     # x = data.drop(class_col, axis=1)
        
#     # y = data[class_col]
#     # x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.8, shuffle=True)

#     Y = data[class_col]
#     X = data.drop(class_col, axis=1)

#     kf = model_selection.KFold(n_splits=n_folds, shuffle=True)

#     for train_index, test_index in kf.split(X):
#         #print("train index", train_index, "test index:", test_index)
#         x_train, x_test = X.loc[train_index, :], X.loc[test_index, :]
#         y_train, y_test = Y.loc(axis=0)[train_index, :], Y.loc(axis=0)[test_index, :]

#         #fitting
#         clf = classifier
#         t_start = process_time()
#         clf.fit(x_train, y_train)
#         t_end = process_time()   
#         fit_diff = t_end - t_start

#         #prediction
#         t_start = process_time()
#         prediction = clf.predict(X=x_test)
#         t_end = process_time()  
#         pred_diff = t_end - t_start

#         #accuracy
#         correct = prediction[prediction == np.array(y_test)]
#         percentage = 100 * len(correct) / len(prediction)
#         print("Procent poprawnie sklasyfikowanych rekordów: {} %".format(percentage))
#         print("Czas trenowania klasyfikatora: {} s".format(fit_diff))
#         print("Czas predykcji klas: {} s".format(pred_diff))

#         labels = ['NO' 'IE' 'ES' 'SE' 'DE' 'NL' 'PL' 'TR' 'BE' 'DK' 'IT' 'FI' 'CZ' 'PT'
#         'CH' 'SI' 'GB' 'FR' 'HR' 'BG' 'GR' 'LT' 'MT' 'IS' 'LV' 'EE' 'CY' 'RO'
#         'HU' 'SK' 'LI' 'LU' 'AT' 'MK']
#         pretty_cm(prediction, y_test, labels)
#         return clf

# def test(clf, x_test):
#     prediction = clf.predict(X=x_test)
#     print(prediction)

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

def logic():

    data = readData()
    print(data.head())
    #data = getLimitedData(data, 0.5)
    #data = data.drop(['Language'], axis=1)


    for classifier_name, classifier in list(dict_classifiers.items()):
        print('Budowanie modelu klasyfikatora ' + classifier_name)
        clf = classification(data, classifier, classifier_name, 'ReceivingCountry')
        filename = classifier_name + ' model'
        pickle.dump(clf, open('model/' + filename + '.sav', 'wb'))
        


    #test(tree_clf, [['27','1','64','2','0','0','14','1827','1299','587']])
    #checkDataValues(data)
    #classification(data)

logic()











# def batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers = 5, verbose = True):
    
#     dict_models = {}
#     for classifier_name, classifier in list(dict_classifiers.items())[:no_classifiers]:
#         t_start = time.clock()
#         classifier.fit(X_train, Y_train)
#         t_end = time.clock()
        
#         t_diff = t_end - t_start
#         train_score = classifier.score(X_train, Y_train)
#         test_score = classifier.score(X_test, Y_test)
        
#         dict_models[classifier_name] = {'model': classifier, 'train_score': train_score, 'test_score': test_score, 'train_time': t_diff}
#         if verbose:
#             print("trained {c} in {f:.2f} s".format(c=classifier_name, f=t_diff))
#     return dict_models


# def display_dict_models(dict_models, sort_by='test_score'):
#     cls = [key for key in dict_models.keys()]
#     test_s = [dict_models[key]['test_score'] for key in cls]
#     training_s = [dict_models[key]['train_score'] for key in cls]
#     training_t = [dict_models[key]['train_time'] for key in cls]
    
#     df_ = pd.DataFrame(data=np.zeros(shape=(len(cls),4)), columns = ['classifier', 'train_score', 'test_score', 'train_time'])
#     for ii in range(0,len(cls)):
#         df_.loc[ii, 'classifier'] = cls[ii]
#         df_.loc[ii, 'train_score'] = training_s[ii]
#         df_.loc[ii, 'test_score'] = test_s[ii]
#         df_.loc[ii, 'train_time'] = training_t[ii]
    
#     display(df_.sort_values(by=sort_by, ascending=False))