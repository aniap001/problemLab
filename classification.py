import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn import model_selection  

from sklearn.decomposition import PCA


from IPython.display import display


import warnings
warnings.filterwarnings("ignore")

def readData():

    data = pd.read_csv('data/encoded.csv', error_bad_lines=False, sep=';')
    return data

dict_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    #"Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000),
    "Decision Tree": tree.DecisionTreeClassifier(),
    #"Random Forest": RandomForestClassifier(n_estimators=1000),
    #"Neural Net": MLPClassifier(alpha = 1),
    "Naive Bayes": GaussianNB(),
}

def batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers = 5, verbose = True):
    
    dict_models = {}
    for classifier_name, classifier in list(dict_classifiers.items())[:no_classifiers]:
        t_start = time.clock()
        classifier.fit(X_train, Y_train)
        t_end = time.clock()
        
        t_diff = t_end - t_start
        train_score = classifier.score(X_train, Y_train)
        test_score = classifier.score(X_test, Y_test)
        
        dict_models[classifier_name] = {'model': classifier, 'train_score': train_score, 'test_score': test_score, 'train_time': t_diff}
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=classifier_name, f=t_diff))
    return dict_models

def checkDataValues(data):
    for col in data.columns.values:
        print(col, data[col].unique())

#funkcja redukująca ilość rekordów w bazie - dla szybszych obliczeń
def getLimitedData(df, ratio):
    mask = np.random.rand(len(df)) < ratio 
    limited_df = df[mask]
    return limited_df

#podziel na zbiory treningowe i testowe
def get_train_test(df, y_col, x_cols, ratio):
    mask = np.random.rand(len(df)) < ratio
    df_train = df[mask]
    df_test = df[~mask]
       
    Y_train = df_train[y_col].values
    Y_test = df_test[y_col].values
    X_train = df_train[x_cols].values
    X_test = df_test[x_cols].values
    #return df_train, df_test, X_train, Y_train, X_test, Y_test
    return X_train, Y_train, X_test, Y_test


def display_dict_models(dict_models, sort_by='test_score'):
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['test_score'] for key in cls]
    training_s = [dict_models[key]['train_score'] for key in cls]
    training_t = [dict_models[key]['train_time'] for key in cls]
    
    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls),4)), columns = ['classifier', 'train_score', 'test_score', 'train_time'])
    for ii in range(0,len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'test_score'] = test_s[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]
    
    display(df_.sort_values(by=sort_by, ascending=False))

 

def classification(data, classifier, class_col, with_reduction):
    #dzielimy zbior danych na dane treningowe i testowe w proporcji 8:2
    x = data.drop(class_col, axis=1)
    if with_reduction:
        x = PCA_reduction(x)
        
    y = data[class_col]
    #x_train, y_train, x_test, y_test = get_train_test(data, x, y, 0.8)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.8)

    #fitting
    clf = classifier
    t_start = time.clock()
    clf.fit(x_train, y_train)
    t_end = time.clock()   
    fit_diff = t_end - t_start

    #prediction
    t_start = time.clock()
    prediction = clf.predict(X=x_test)
    t_end = time.clock()  
    pred_diff = t_end - t_start

    #accuracy
    correct = prediction[prediction == np.array(y_test)]
    percentage = 100 * len(correct) / len(prediction)
    print("Procent poprawnie sklasyfikowanych rekordów: {} %".format(percentage))
    print("Czas trenowania klasyfikatora: {} s".format(fit_diff))
    print("Czas predykcji klas: {} s".format(pred_diff))
    return clf

def PCA_reduction(x):
    pca = PCA(n_components=5).fit(x)
    x = pca.transform(x)
    return x

def test(clf, x_test):
    prediction = clf.predict(X=x_test)
    print(prediction)


def logic():

    data = readData()
    #print(data.dtypes)
    #data = getLimitedData(data, 0.5)

    #tree_clf = classification(data, tree.DecisionTreeClassifier(), 'ReceivingCountry', False)
    #knn_clf = classification(data, GaussianNB(), 'ReceivingCountry', False)

    for classifier_name, classifier in list(dict_classifiers.items()):
        print('Budowanie modelu klasyfikatora ' + classifier_name)
        classification(data, classifier, 'ReceivingCountry', False)


    #test(tree_clf, [['27','1','64','2','0','0','14','1827','1299','587']])
    #display_dict_models
    #checkDataValues(data)
    #classification(data)

logic()