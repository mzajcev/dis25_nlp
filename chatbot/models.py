import joblib
from .chat_logger import log_and_print


def load_models():
    loaded_clf_logistic = joblib.load('classifiers/logistic_regression/classifier.logistic_regression')
    loaded_vectorizer_logistic = joblib.load('classifiers/logistic_regression/vectors.logistic_regression')
    loaded_clf_naive = joblib.load('classifiers/naive_bayes/classifier.naive_bayes')
    loaded_vectorizer_naive = joblib.load('classifiers/naive_bayes/vectors.naive_bayes')

    return loaded_clf_logistic, loaded_clf_naive, loaded_vectorizer_logistic, loaded_vectorizer_naive


loaded_clf_logistic, loaded_clf_naive, loaded_vectorizer_logistic, loaded_vectorizer_naive = load_models()

def classify_sentence(sentence, technique):
    global loaded_clf_logistic, loaded_clf_naive, loaded_vectorizer_logistic, loaded_vectorizer_naive

    if technique == 'Logistic Regression':
        vector = loaded_vectorizer_logistic
        classifier = loaded_clf_logistic
    elif technique == 'Naive Bayes':
        vector = loaded_vectorizer_naive
        classifier = loaded_clf_naive
    else:
        log_and_print("Chatbot", "Invalid technique. Please choose either 'Logistic Regression' or 'Naive Bayes'.")
        return

    vectorized_sentence = vector.transform([sentence])
    classification_result = classifier.predict(vectorized_sentence)

        # Convert the numeric classification result to "negative" or "positive"
    if classification_result[0] == 0:
        classification_label = "negative"
    elif classification_result[0] == 1:
        classification_label = "positive"
    else:
        classification_label = "unknown"

    response = f"The result of the {technique} classification is: {classification_label}"
    log_and_print('Chatbot', response)
