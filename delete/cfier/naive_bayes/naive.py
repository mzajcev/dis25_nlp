import re
import string
import pandas as pd
import nltk
import tensorflow_datasets as tfds
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from string import punctuation
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data():
    data = tfds.load('imdb_reviews', split={'train': 'train', 'test': 'test'})
    train_df = tfds.as_dataframe(data['train'])
    test_df = tfds.as_dataframe(data['test'])

    train_df['text'] = train_df['text'].apply(lambda x: x.decode('utf-8'))
    test_df['text'] = test_df['text'].apply(lambda x: x.decode('utf-8'))

    train_df['label'] = train_df['label'].replace({0: 'negative', 1: 'positive'})
    test_df['label'] = test_df['label'].replace({0: 'negative', 1: 'positive'})

    return train_df, test_df

def prepare_data(train_df, test_df):
    x_train, x_test, y_train, y_test = train_test_split(train_df, train_df['label'], test_size=0.2, random_state=42, stratify=train_df['label'])

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    return x_train, x_test, y_train, y_test

def transformations(dataframe):
    lemmatizer = WordNetLemmatizer()
    cachedStopWords = stopwords.words("english")

    dataframe['text'] = dataframe['text'].apply(lambda words: re.sub('<[^<]+?>', '', words))
    dataframe['text'] = dataframe['text'].apply(lambda words: words.lower())
    dataframe['text'] = dataframe['text'].apply(word_tokenize)
    dataframe['text'] = dataframe['text'].apply(lambda words: [x for x in words if not x in punctuation])
    dataframe['text'] = dataframe['text'].apply(lambda words: [x for x in words if not x.isdigit()])
    dataframe['text'] = dataframe['text'].apply(lambda words: [x for x in words if x not in cachedStopWords])
    dataframe['text'] = dataframe['text'].apply(lambda words: [lemmatizer.lemmatize(x) for x in words])
    dataframe['text'] = dataframe['text'].apply(lambda words: " ".join(words))

    return dataframe

def vectorize_data(x_train, x_test):
    vec = CountVectorizer()
    vec = vec.fit(x_train.text)
    train_x_bow = vec.transform(x_train.text)
    test_x_bow = vec.transform(x_test.text)

    return train_x_bow, test_x_bow

def tune_model(train_x_bow, y_train):
    alpha_ranges = {
        "alpha": [0.001, 0.01, 0.1, 1, 10.0, 100]
    }

    classifier = MultinomialNB()
    grid_search = GridSearchCV(classifier, param_grid=alpha_ranges, scoring='accuracy', cv=2, return_train_score=True)
    grid_search.fit(train_x_bow, y_train)

    alpha = [0.001, 0.01, 0.1, 1, 10.0, 100]
    train_acc = grid_search.cv_results_['mean_train_score']
    test_acc = grid_search.cv_results_['mean_test_score']

    plt.plot(alpha, train_acc, label="Training Score", color='b')
    plt.plot(alpha, test_acc, label="Cross Validation Score", color='r')
    plt.title("Validation Curve with Naive Bayes Classifier")
    plt.xlabel("Alpha")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc = 'best')
    plt.show()

    return grid_search.best_estimator_

def evaluate_model(best_classifier, train_x_bow, y_train, test_x_bow, y_test):
    best_classifier.fit(train_x_bow, y_train)
    predict = best_classifier.predict(test_x_bow)
    print("Accuracy is ", accuracy_score(y_test, predict))
    print("Report: ", classification_report(y_test, predict))

if __name__ == "__main__":
    train_df, test_df = load_data()
    x_train, x_test, y_train, y_test = prepare_data(train_df, test_df)
    x_train = transformations(x_train)
    x_test = transformations(x_test)
    train_x_bow, test_x_bow = vectorize_data(x_train, x_test)
    best_classifier = tune_model(train_x_bow, y_train)
    evaluate_model(best_classifier, train_x_bow, y_train, test_x_bow, y_test)
