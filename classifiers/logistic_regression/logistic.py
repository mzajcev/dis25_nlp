import re
import pandas as pd
import nltk
import tensorflow_datasets as tfds
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

nltk.download('punkt')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
cachedStopWords = stopwords.words("english")
exclude_stopword = {'not', 'against', 'nor', 'no'}
stop_words = ([word for word in cachedStopWords if word not in exclude_stopword])


def load_data():
    # Load the IMDb reviews dataset
    data = tfds.load('imdb_reviews', split={'train': 'train', 'test': 'test'})
    
    # Convert the data into pandas DataFrame and decode bytes to string
    train_df = tfds.as_dataframe(data['train'])
    test_df = tfds.as_dataframe(data['test'])

    train_df['text'] = train_df['text'].apply(lambda x: x.decode('utf-8'))
    test_df['text'] = test_df['text'].apply(lambda x: x.decode('utf-8'))

    train_df['label'] = train_df['label'].replace({0: 'negative', 1: 'positive'})
    test_df['label'] = test_df['label'].replace({0: 'negative', 1: 'positive'})

    return train_df, test_df

def preprocess_data(dataframe):
    # HTML Tags removal
    dataframe['text'] = dataframe['text'].apply(lambda words: re.sub('<[^<]+?>', '', words))

    # Lower case conversion
    dataframe['text'] = dataframe['text'].apply(lambda words: words.lower())

    # Word Tokenization
    dataframe['text'] = dataframe['text'].apply(word_tokenize)

    # Punctuation removal
    dataframe['text'] = dataframe['text'].apply(lambda words: [x for x in words if not x in punctuation])

    # Number removal
    dataframe['text'] = dataframe['text'].apply(lambda words: [x for x in words if not x.isdigit()])

    # Stopword removal
    dataframe['text'] = dataframe['text'].apply(lambda words: [x for x in words if x not in stop_words])

    # Lemmatization
    dataframe['text'] = dataframe['text'].apply(lambda words: [lemmatizer.lemmatize(x) for x in words])

    # Join again
    dataframe['text'] = dataframe['text'].apply(lambda words: " ".join(words))

    return dataframe

def model_training(x_train, y_train):
    vec = CountVectorizer()
    vec = vec.fit(x_train.text)
    train_x_bow = vec.transform(x_train.text)

    # Create a Logistic Regression classifier
    classifier = LogisticRegression(max_iter=100)

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10.0, 100]  # Replace this list with your 'alpha_ranges' values for 'C'
    }

    # Create GridSearchCV object
    grid_search = GridSearchCV(classifier, param_grid=param_grid, scoring='accuracy', cv=2, return_train_score=True)

    # Fit the model on the training data
    grid_search.fit(train_x_bow, y_train)

    # Get the best estimator from the grid search
    best_classifier = grid_search.best_estimator_

    return best_classifier, vec

def evaluate_model(classifier, vec, x_test, y_test):
    test_x_bow = vec.transform(x_test.text)
    predict = classifier.predict(test_x_bow)
    print("Accuracy is ", accuracy_score(y_test, predict))
    print("Report: ", classification_report(y_test, predict))

def main():
    train_df, test_df = load_data()

    le = LabelEncoder()
    y_train = le.fit_transform(train_df['label'])
    y_test = le.transform(test_df['label'])

    x_train = preprocess_data(train_df)
    x_test = preprocess_data(test_df)

    best_classifier, vec = model_training(x_train, y_train)

    evaluate_model(best_classifier, vec, x_test, y_test)

if __name__ == "__main__":
    main()
