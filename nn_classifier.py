import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
from sklearn.naive_bayes import GaussianNB

def load_dataset(path):
    df = pd.read_csv(path,
                 sep=';',
                 encoding='LATIN1',
                 decimal=',')
    np_df = df.values
    features = np_df[:,:-1]
    label = np_df[:,-1]
    label = label.reshape((len(label), 1))
    return(features, label)

def str_cualitative_features(X):
    X[:,:4] = X[:,:4].astype(str)
    X[:,4:] = X[:,4:].astype(float)
    return(X)

def prepare_inputs(X):
    oe = OrdinalEncoder()
    oe.fit(X[:,:4])
    X[:,:4] = oe.transform(X[:,:4])
    return(X)

def prepare_targets(y):
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    return(y)

def define_nn(X_train):
    model = Sequential()
    model.add(Dense (10, input_dim = X_train.shape[1], activation = 'relu', kernel_initializer = 'he_normal'))
    model.add(Dense (1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return(model)

def keras_precision(X_test, y_test):
    _, precision = model.evaluate(X_test, y_test, verbose = 0)
    print ('precision:% .2f'% (precision * 100)) 

def get_confusion_matrix(X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = [1 if i>0.5 else 0 for i in y_pred]
    print(confusion_matrix(y_test, y_pred))
    
    
X, y = load_dataset(path='./deudas.csv')
X = str_cualitative_features(X)
X = prepare_inputs(X)
y = prepare_targets(y)

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.33, random_state = 1)

model = define_nn(X_train)
model.fit(X_train, y_train, epochs = 100, batch_size = 16, verbose = 2)
keras_precision(X_test, y_test)
get_confusion_matrix(X_test, y_test)

#Naive bayes classifier
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))





