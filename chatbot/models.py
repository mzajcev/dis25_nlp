import joblib

from . import recommendations
from .chat_logger import log_and_print

# Function to load logistic regression and naive bayes model and their vectorizers
def load_models():
    loaded_clf_logistic = joblib.load('./classifiers/logistic_regression/classifier.logistic_regression')
    loaded_vectorizer_logistic = joblib.load('./classifiers/logistic_regression/vectors.logistic_regression')

    loaded_clf_naive = joblib.load('./classifiers/naive_bayes/classifier.naive_bayes')
    loaded_vectorizer_naive = joblib.load('./classifiers/naive_bayes/vectors.naive_bayes')

    return loaded_clf_logistic, loaded_clf_naive, loaded_vectorizer_logistic, loaded_vectorizer_naive


# Call the 'load_models' function
loaded_clf_logistic, loaded_clf_naive, loaded_vectorizer_logistic, loaded_vectorizer_naive = load_models()

# Function to classify a sentence using logistic regression or naive bayes
def classify_sentence(sentence, technique):
    global loaded_clf_logistic, loaded_clf_naive, loaded_vectorizer_logistic, loaded_vectorizer_naive

    # Depending on user input, select the matching classifier and vectorizer
    if technique == 'Logistic Regression':
        vector = loaded_vectorizer_logistic
        classifier = loaded_clf_logistic
    elif technique == 'Naive Bayes':
        vector = loaded_vectorizer_naive
        classifier = loaded_clf_naive
    else:
        # If technique is not recognized, print error message and return
        log_and_print("Chatbot", "Invalid technique. Please choose either 'Logistic Regression' or 'Naive Bayes'.")
        return

    # Vectorize the sentence with the chosen vectorizer
    vectorized_sentence = vector.transform([sentence])
    # Classify the vectorized sentence with the chosen classifier
    classification_result = classifier.predict(vectorized_sentence)

    # Convert the classification result to "NEGATIVE" or "POSITIVE"
    if classification_result[0] == "negative":
        classification_label = "NEGATIVE"
    elif classification_result[0] == "positive":
        classification_label = "POSITIVE"
    else:
        classification_label = "unknown"

    # Create response containing the classification result
    response = f"The result of the {technique} classification is: {classification_label}\n"
    # Use the recommendations module to recommend content based on the classification result and add to response
    recommendation = recommendations.recommend_content(classification_label)
    response += f"Based on your sentiment analysis result, here is a recommendation for you: {recommendation}"
    # Log and print the response
    log_and_print('Chatbot', response)
