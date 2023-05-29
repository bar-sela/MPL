import numpy
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from backPropagation import *


def preProcceing(X_train, X_test):
    scaler = StandardScaler()  # this scaling ensures that the features have similar ranges and helps in achieving better model performance
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled , X_test_scaled

def get_x_y(chars:list):
    with open("./result.txt", "r") as f:
        lines = f.readlines()  # read all lines from file
    vectors = []
    Y = []
    for line in lines:
        line = line.replace("(", "").replace(")", "")
        label = line[0]
        if label in chars:
            label = 0 if label == chars[0] else 1
            Y.append(int(label))
            # split the line by whitespace and convert values to float
            values = list(map(np.float64, line.split(',')[1:]))
            # append the values as a numpy array to the vectors list
            vectors.append(np.array(values))

    # stack all vectors into a 2D numpy array
    X = np.vstack(vectors)
    Y = np.array(Y)

    return X, Y

def main(chars:list, lr, stop, debug=False):
        _X, _Y = get_x_y(chars)
        # using the train test split function
        X_train, X_test, y_train, y_test = train_test_split(_X, _Y, test_size=0.2, random_state=42)
        X_train_fit, X_val, y_train_fit, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        # create model
        model = Model()
        # train - prints acc and var of each fold
        train_with_crossValidation(model,X_train, y_train , X_train_fit, y_train_fit )
        # fit
        # Predict the labels for the testing data
        y_pred=  predict(model,X_test)
        #cheaking the results
        accuracy = acc(y_test, y_pred)

        print(f"\t->Final accuracy on test set: {accuracy} " )

if __name__ == "__main__":
    print("\n-------------------------- mem & bet ---------------------------------------")
    main(['2', '1'], 0.0001, 130, debug=False)  # Best: 0.0001, 130 -> test-0.84
    print("-------------------------- bet & lamed ---------------------------------------")
    main(['1', '3'], 0.0001, 130, debug=False)  # Best: 0.0001, 130 -> test-0.84
    print("\n-------------------------- mem & lamed ---------------------------------------")
    main(['2', '3'], 0.0001, 130, debug=False)  # Best: 0.0001, 130 -> test-0.84

