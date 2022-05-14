import nltk 
import re
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from wordcloud import WordCloud

vocab_size = 5000

def preprocess(train_data, test_data):

    train_data = train_data.fillna('')
    test_data = test_data.fillna('')

    # build_wordcloud(train_data)

    train_data['content'] = train_data['title'] + ' ' + train_data['author']
    test_data['content'] = test_data['title'] + ' ' + test_data['author']


    X = train_data.drop('label', axis=1)
    Y = train_data['label']

    nltk.download('stopwords')

    corpus_train, corpus_test = stem(X), stem(test_data)
    onehot_train, onehot_test = [one_hot(words, vocab_size) for words in corpus_train], [one_hot(words, vocab_size) for words in corpus_test]
    padded_train, padded_test = pad_sequences(onehot_train, padding='pre', maxlen=25), pad_sequences(onehot_test, padding='pre', maxlen=25)
    
    X, Y, test_data = np.array(padded_train), np.array(Y), np.array(padded_test)
    return X, Y, test_data
    

def stem(dataset):
    ps = PorterStemmer()
    corpus = []

    for i in range(len(dataset)):
        seq = re.sub('[^a-zA-Z]',' ', dataset['content'][i])
        seq = seq.lower()
        seq = seq.split()
        seq = [ps.stem(word) for word in seq if not word in stopwords.words('english')]
        seq = ' '.join(seq)
        corpus.append(seq)

    return corpus


def build_wordcloud(dataset):
    wordcloud = WordCloud(background_color='black', width=500, height=500)
    textcloud = wordcloud.generate(' '.join(dataset[dataset['label']==1]['text']))
    plt.figure(figsize=(20, 30))
    plt.imshow(textcloud)
    plt.axis('off')
    plt.show()



