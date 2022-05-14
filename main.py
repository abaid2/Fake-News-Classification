from webbrowser import BackgroundBrowser
from preprocess import preprocess
from model import build_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

def test_model(test, train):
    # Check if the files exist
    if not os.path.exists(test):
        print('The file {} does not exist'.format(test))
        exit()
    elif not os.path.exists(train):
        print('The file {} does not exist'.format(train))
        exit()

    train_data = pd.read_csv(train)
    test_data = pd.read_csv(test)

    X, Y, test_data = preprocess(train_data, test_data)

    x_train, X_test, y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42, stratify=Y)
    X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
    X_test = test_data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)

    model = build_model()
    history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=10, batch_size=64) 
    plot_graphs(history)   

    predictions = np.round(model.predict(X_test).flatten()).astype('int')

    print(classification_report(Y_test, predictions))
    print(accuracy_score(Y_test, predictions))


def plot_graphs(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()









def main():
    test = 'data/test.csv'
    train = 'data/train.csv'
    test_model(test, train)
    pass



if __name__ == "__main__":
    main()
    