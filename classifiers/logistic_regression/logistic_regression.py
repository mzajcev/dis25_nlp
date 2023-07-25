import re
import pandas as pd
import tensorflow_datasets as tfds
from joblib import dump
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from sklearn.metrics import classification_report

# Function to load the data
def load_data():
    data = tfds.load('imdb_reviews', split={'train': 'train', 'test': 'test'})
    train_df = tfds.as_dataframe(data['train'])
    test_df = tfds.as_dataframe(data['test'])
    train_df['text'] = train_df['text'].apply(lambda x: x.decode('utf-8'))
    test_df['text'] = test_df['text'].apply(lambda x: x.decode('utf-8'))
    train_df['label'] = train_df['label'].replace({0: 'negative', 1: 'positive'})
    test_df['label'] = test_df['label'].replace({0: 'negative', 1: 'positive'})
    return train_df, test_df

# Define your preprocessing function
lemmatizer = WordNetLemmatizer()
cachedStopWords = stopwords.words("english")
exclude_stopword = {'not', 'against', 'nor', 'no'}
stop_words = ([word for word in cachedStopWords if word not in exclude_stopword])

def preprocess_data(dataframe):
    dataframe['text'] = dataframe['text'].apply(lambda words: re.sub('<[^<]+?>', '', words))
    dataframe['text'] = dataframe['text'].apply(lambda words: words.lower())
    dataframe['text'] = dataframe['text'].apply(word_tokenize)
    dataframe['text'] = dataframe['text'].apply(lambda words: [x for x in words if not x in punctuation])
    dataframe['text'] = dataframe['text'].apply(lambda words: [x for x in words if not x.isdigit()])
    dataframe['text'] = dataframe['text'].apply(lambda words: [x for x in words if x not in stop_words])
    dataframe['text'] = dataframe['text'].apply(lambda words: [lemmatizer.lemmatize(x) for x in words])
    dataframe['text'] = dataframe['text'].apply(lambda words: " ".join(words))
    return dataframe

# Function for model training
def model_training(x_train, y_train):
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1,2)) # added bigrams
    vec.fit(x_train)
    train_x_bow = vec.transform(x_train)

    model = LogisticRegression(max_iter=1000, penalty='l2')  
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}  

    grid_search = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', cv=5, return_train_score=True)
    grid_search.fit(train_x_bow, y_train)
    best_model = grid_search.best_estimator_

    print("\nCross-validation Results:")
    for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
        print(f"Mean CV accuracy: {mean_score:.4f} - Params: {params}")

    return best_model, vec

# Load the data
train_df, test_df = load_data()

# Preprocess the data
train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)

# Logistic Regression
lr_model, vec_lr = model_training(train_df['text'], train_df['label']) 

# Vectorize the test data
X_test_lr = vec_lr.transform(test_df['text'])  

# Map labels back to binary
y_test = test_df['label'].map({'negative': 0, 'positive': 1})

# Evaluate the model on test set
y_pred_lr = lr_model.predict(X_test_lr)  
if isinstance(y_pred_lr[0], str):
    y_pred_lr = pd.Series(y_pred_lr).map({'negative': 0, 'positive': 1}).values
print(f'Logistic Regression Test Accuracy: {accuracy_score(y_test, y_pred_lr)}')  

# Print classification report
print('Classification Report:')
print(classification_report(y_test, y_pred_lr, target_names=['Negative', 'Positive']))

# dump(vec_lr, 'vectors.logistic_regression')
# dump(lr_model, 'classifier.logistic_regression')

# Cross-validation Results:
# Mean CV accuracy: 0.8176 - Params: {'C': 0.001}
# Mean CV accuracy: 0.8286 - Params: {'C': 0.01}
# Mean CV accuracy: 0.8643 - Params: {'C': 0.1}
# Mean CV accuracy: 0.8878 - Params: {'C': 1}
# Mean CV accuracy: 0.8866 - Params: {'C': 10}
# Mean CV accuracy: 0.8706 - Params: {'C': 100}
# Mean CV accuracy: 0.8599 - Params: {'C': 1000}
# Logistic Regression Test Accuracy: 0.88692

# Classification Report:
#               precision    recall  f1-score   support

#     Negative       0.89      0.89      0.89     12500
#     Positive       0.89      0.89      0.89     12500

#     accuracy                           0.89     25000
#    macro avg       0.89      0.89      0.89     25000
# weighted avg       0.89      0.89      0.89     25000