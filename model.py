import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential

vocab_size = 5000

def build_model():
    model = Sequential(
    [
        Embedding(vocab_size, 40, input_length=25),
        Dropout(0.3), 
        LSTM(100),
        Dropout(0.3),
        Dense(64,activation='relu'),
        Dropout(0.3),
        Dense(1,activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
