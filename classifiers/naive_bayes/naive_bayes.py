import re
import joblib
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from sklearn.metrics import classification_report


# Function to load the data
def load_data():
    # Load the IMDb reviews dataset
    data = tfds.load('imdb_reviews', split={'train': 'train', 'test': 'test'})
    
    # Convert the data into pandas DataFrame and decode bytes to string
    train_df = tfds.as_dataframe(data['train'])
    test_df = tfds.as_dataframe(data['test'])

    train_df['text'] = train_df['text'].apply(lambda x: x.decode('utf-8'))
    test_df['text'] = test_df['text'].apply(lambda x: x.decode('utf-8'))


    return train_df, test_df

# Load the data
train_df, test_df = load_data()

# Defining of preprocessing function
lemmatizer = WordNetLemmatizer()
cachedStopWords = stopwords.words("english")
exclude_stopword = {'not', 'against', 'nor', 'no'}
stop_words = ([word for word in cachedStopWords if word not in exclude_stopword])

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


# Define your training function for Naive Bayes
def model_training_naive(x_train, y_train):
    vec = TfidfVectorizer(max_features=10000)
    vec.fit(x_train)
    train_x_bow = vec.transform(x_train)

    model = MultinomialNB()
    param_grid = {'alpha': [0.5]}

    grid_search = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', cv=5, return_train_score=True)
    grid_search.fit(train_x_bow, y_train)
    best_model = grid_search.best_estimator_

    print("\nCross-validation Results:")
    for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
        print(f"Mean CV accuracy: {mean_score:.4f} - Params: {params}")

    return best_model, vec


# Preprocess the data
train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)


# Prepare the training data and labels
x_train = train_df['text']
y_train = train_df['label']


# Train the Naive Bayes model
naive_model, naive_vec = model_training_naive(x_train, y_train)

# Prepare the testing data and labels
x_test = test_df['text']
y_test = test_df['label']

# Transform the testing data using the vectorizer from the naive model
x_test_naive = naive_vec.transform(x_test)

# Predict the labels for the test set using the naive model
y_pred_naive = naive_model.predict(x_test_naive)

# Calculate and print the accuracy and classification report for the naive model
print("\nNaive Bayes Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_naive)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_naive)}")

# Naive Bayes Results:
# Accuracy: 0.83392
# Classification Report:
#               precision    recall  f1-score   support

#            0       0.81      0.87      0.84     12500
#            1       0.86      0.80      0.83     12500

#     accuracy                           0.83     25000
#    macro avg       0.84      0.83      0.83     25000
# weighted avg       0.84      0.83      0.83     25000

# joblib.dump(naive_vec, 'vectors.naive_bayes')
# joblib.dump(naive_model, 'classifier.naive_bayes')