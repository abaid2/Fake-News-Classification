from preprocess import preprocess
from model import build_model

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
    model = build_model()
    model.fit(X, Y, epochs=20, batch_size=64)



def main():
    test = 'data/test.csv'
    train = 'data/train.csv'
    test_model(test, train)
    pass


if __name__ == "__main__":
    main()
    