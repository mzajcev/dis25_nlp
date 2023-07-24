import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from spellchecker import SpellChecker
import joblib

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
spell = SpellChecker()


loaded_clf_logistic = joblib.load('classifier.logistic_regression')
loaded_vectorizer_logistic = joblib.load('vectors.logistic_regression')

loaded_clf_naive = joblib.load('classifier.naive_bayes')
loaded_vectorizer_naive = joblib.load('vectors.naive_bayes')


def clean_user_input(user_input):
    user_input_no_html = re.sub('<.*?>', '', user_input)
    user_input_token = word_tokenize(user_input_no_html)
    user_input_punct_lower = [x.lower() for x in user_input_token if x not in punctuation]
    user_input_no_num = [x for x in user_input_punct_lower if not x.isdigit()]
    correct_words = [spell.correction(word) for word in user_input_no_num]
    filtered_words = [word for word in correct_words if word not in stop_words]
    base_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    clean_response = ' '.join(base_words)
    return clean_response

def classify_sentence(sentence, technique):
    if technique == 'logistic':
        vector = loaded_vectorizer_logistic
        classifier = loaded_clf_logistic
    elif technique == 'naive':
        vector = loaded_vectorizer_naive
        classifier = loaded_clf_naive
    else:
        print("Invalid technique. Please choose either 'logistic' or 'naive'.")
        return

   
    vectorized_sentence = vector.transform([sentence])
    classification_result = classifier.predict(vectorized_sentence)
    print(f"The result of the {technique} classification is: {classification_result[0]}")

def main():
    user_input = input("Please enter the sentence you want to analyze: ")
    cleaned_sentence = clean_user_input(user_input)
    print("Cleaned sentence: ", cleaned_sentence)

    technique = input("Which technique would you like to use (logistic/naive)? ")
    classify_sentence(cleaned_sentence, technique)

if __name__ == "__main__":
    main()