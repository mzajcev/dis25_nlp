import joblib
import re
import re
import nltk
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer
from string import punctuation
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from googletrans import Translator

def load_models():
    loaded_clf_logistic = joblib.load('../classifiers/logistic_regression/classifier.logistic_regression')
    loaded_vectorizer_logistic = joblib.load('../classifiers/logistic_regression/vectors.logistic_regression')

    loaded_clf_naive = joblib.load('../classifiers/naive_bayes/classifier.naive_bayes')
    loaded_vectorizer_naive = joblib.load('../classifiers/naive_bayes/vectors.naive_bayes')

    return loaded_clf_logistic, loaded_clf_naive, loaded_vectorizer_logistic, loaded_vectorizer_naive


# Call the 'load_models' function
loaded_clf_logistic, loaded_clf_naive, loaded_vectorizer_logistic, loaded_vectorizer_naive = load_models()

def classify_sentence(sentence, technique):
    global loaded_clf_logistic, loaded_clf_naive, loaded_vectorizer_logistic, loaded_vectorizer_naive

    # Depending on user input, select the matching classifier and vectorizer
    if technique == 'logistic':
        vector = loaded_vectorizer_logistic
        classifier = loaded_clf_logistic
    elif technique == 'naive':
        vector = loaded_vectorizer_naive
        classifier = loaded_clf_naive
    else:
        # If technique is not recognized, print error message and return
        print("Error: technique not recognized")
        return

    # Vectorize the sentence with the chosen vectorizer
    vectorized_sentence = vector.transform([sentence])
    # Classify the vectorized sentence with the chosen classifier
    classification_result = classifier.predict(vectorized_sentence)

    # Convert the classification result to "NEGATIVE" or "POSITIVE"
    if classification_result[0] == 0:
        classification_label = "NEGATIVE"
        return
    elif classification_result[0] == 1:
        classification_label = "POSITIVE"
        return
    else:
        classification_label = "unknown"

    # Create response containing the classification result
    response = f"The result of the {technique} classification is: {classification_label}\n"


# Initialize lemmatizer, spell checker, and translator
lemmatizer = WordNetLemmatizer()
spell = SpellChecker()
translator = Translator()

# Define stop words and exclude 'not', 'against', 'nor', 'no' from the list
stop_words_orig = set(stopwords.words('english'))
exclude_stopword = {'not', 'against', 'nor', 'no'}
stop_words = ([word for word in stop_words_orig if word not in exclude_stopword])

# Function clean user input
def clean_user_input(user_input):
    # Remove HTML tags
    user_input_no_html = re.sub('<.*?>', '', user_input)
    
    # Translate user input to english
    user_input_en = translator.translate(user_input_no_html, dest='en').text
    
    # Tokenize the translated input
    user_input_token = word_tokenize(user_input_en)
    
    # Remove punctuation and make user input lower case
    user_input_punct_lower = [x.lower() for x in user_input_token if x not in punctuation]
    
    # Remove numbers
    user_input_no_num = [x for x in user_input_punct_lower if not x.isdigit()]
    
    # Correct misspelled words
    correct_words = [spell.correction(word) for word in user_input_no_num]
    
    # Remove stop words and None values
    filtered_words = [word for word in correct_words if word not in stop_words and word is not None]
    
    # Apply lemmatization
    base_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    
    # Join words back into sentence
    clean_response = ' '.join(base_words)
    
    # Return cleaned response
    return clean_response

def load_sentences_from_file(file_path):
    with open('test_data.csv', 'r') as file:
        sentences = file.read().splitlines()
    return sentences

def process_sentences(sentences, technique):
    for sentence in sentences:
        cleaned_sentence = clean_user_input(sentence)
        classify_sentence(cleaned_sentence, technique)

def main():
    file_path = "test_data.csv"  # replace with the path to your text file
    sentences = load_sentences_from_file(file_path)

    # Classify sentences using naive bayes technique
    print("\nClassifying sentences using naive bayes technique...")
    process_sentences(sentences, 'naive')

    # Classify sentences using logistic regression technique
    print("\nClassifying sentences using logistic regression technique...")
    process_sentences(sentences, 'logistic')

if __name__ == "__main__":
        main()
