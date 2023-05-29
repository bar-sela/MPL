import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

file1 = 'result.txt'  # loading the file with all the vectors represents the letters ב,ל,מ


# This function will use us to read and create DataFrame from the vectors
def createDF(file_name):
    data = []
    with open(file_name, 'r', encoding='latin-1') as f:
        file_content = f.read()
    lines = file_content.splitlines()
    for line in lines:
        values = line.strip().strip('()').replace('[', '').replace(']', '').split(',')
        data.append(values)

    dataframe = pd.DataFrame(data)  # pandas function
    return dataframe


""" 
    Create and train the MLP classifier.
    MLP classifier is a class from scikit-learn that represents a multi-layer perceptron (MLP) classifier, 
    which is a type of artificial neural network.
    @:param "hidden_layer_sizes" is specifies the architecture of the MLP, in our case we creates four hidden layers, each with 100 neurons.
    @:param "activation" specifies the activation function to be used in the hidden layers. 'relu' helps to capture complex patterns in the data.
    @:param "solver" is specifies the optimization algorithm used for training the MLP.
    'adam' adjusts the learning rate adaptively during training and it is good for us.
"""
def Model():
    mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100), activation='relu', solver='adam', random_state=42)
    return mlp
def train_with_crossValidation(mlp , X, y, X_train, y_train):
    scores = cross_val_score(mlp, X, y, cv=5)
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    for i in range(0, 5):
        print(f" Fold {i} acc: {scores[i]}  ")
    print(scores)
    print("Mean Validation Score:", mean_score)
    print("Standard Deviation of Validation Scores:", std_score)
    fit(mlp,X_train, y_train)

def fit(mlp, X_train,y_train):
    mlp.fit(X_train, y_train)
def predict(mlp,X_test ):
    y_pred = mlp.predict(X_test)
    return y_pred
def acc(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    return  accuracy








