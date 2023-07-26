import joblib
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
    loaded_clf_logistic = joblib.load('classifier.logistic_regression')
    loaded_vectorizer_logistic = joblib.load('vectors.logistic_regression')

    loaded_clf_naive = joblib.load('classifier.naive_bayes')
    loaded_vectorizer_naive = joblib.load('vectors.naive_bayes')

    return loaded_clf_logistic, loaded_clf_naive, loaded_vectorizer_logistic, loaded_vectorizer_naive


# Call the 'load_models' function
loaded_clf_logistic, loaded_clf_naive, loaded_vectorizer_logistic, loaded_vectorizer_naive = load_models()




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

def classify_sentence(sentence, technique):
    global loaded_clf_logistic, loaded_clf_naive, loaded_vectorizer_logistic, loaded_vectorizer_naive

    if technique == 'logistic':
        vector = loaded_vectorizer_logistic
        classifier = loaded_clf_logistic
    elif technique == 'naive':
        vector = loaded_vectorizer_naive
        classifier = loaded_clf_naive
    else:

        print("Error: technique not recognized")
        return


    vectorized_sentence = vector.transform([sentence])

    classification_result = classifier.predict(vectorized_sentence)


    if classification_result[0] == 0:
        classification_label = "NEGATIVE"
    elif classification_result[0] == 1:
        classification_label = "POSITIVE"
    else:
        classification_label = "unknown"

 
    response = f"The result of the {technique} classification is: {classification_label}\n"
    return response 


def process_sentences(sentences, technique):
    for sentence in sentences:
        cleaned_sentence = clean_user_input(sentence)
        print(classify_sentence(cleaned_sentence, technique))

def main():
    file_path = "test_data.csv" 
    sentences = load_sentences_from_file(file_path)


    print("\nClassifying sentences using naive bayes technique...")
    process_sentences(sentences, 'naive')


    print("\nClassifying sentences using logistic regression technique...")
    process_sentences(sentences, 'logistic')


    

# if __name__ == "__main__":
#         main()


# Classifying sentences using naive bayes technique...
# The result of the naive classification is: NEGATIVE
# The result of the naive classification is: POSITIVE
# The result of the naive classification is: POSITIVE
# The result of the naive classification is: NEGATIVE
# The result of the naive classification is: POSITIVE
# The result of the naive classification is: NEGATIVE
# The result of the naive classification is: POSITIVE
# The result of the naive classification is: NEGATIVE
# The result of the naive classification is: NEGATIVE
# The result of the naive classification is: POSITIVE
# The result of the naive classification is: NEGATIVE
# The result of the naive classification is: POSITIVE
# The result of the naive classification is: NEGATIVE
# The result of the naive classification is: NEGATIVE
# The result of the naive classification is: POSITIVE
# The result of the naive classification is: POSITIVE
# The result of the naive classification is: POSITIVE
# The result of the naive classification is: NEGATIVE
# The result of the naive classification is: NEGATIVE
# The result of the naive classification is: NEGATIVE
# The result of the naive classification is: POSITIVE
# The result of the naive classification is: POSITIVE
# The result of the naive classification is: NEGATIVE
# The result of the naive classification is: POSITIVE
# The result of the naive classification is: POSITIVE
# The result of the naive classification is: POSITIVE
# The result of the naive classification is: NEGATIVE
# The result of the naive classification is: NEGATIVE
# The result of the naive classification is: NEGATIVE
# The result of the naive classification is: POSITIVE
# The result of the naive classification is: POSITIVE
# The result of the naive classification is: POSITIVE
# The result of the naive classification is: NEGATIVE
# The result of the naive classification is: NEGATIVE
# The result of the naive classification is: POSITIVE
# The result of the naive classification is: POSITIVE
# The result of the naive classification is: NEGATIVE
# The result of the naive classification is: NEGATIVE
# The result of the naive classification is: POSITIVE
# The result of the naive classification is: POSITIVE
# The result of the naive classification is: NEGATIVE
# The result of the naive classification is: NEGATIVE
# The result of the naive classification is: POSITIVE
# The result of the naive classification is: POSITIVE
# The result of the naive classification is: POSITIVE
# The result of the naive classification is: NEGATIVE
# The result of the naive classification is: POSITIVE
# The result of the naive classification is: NEGATIVE
# The result of the naive classification is: POSITIVE
# The result of the naive classification is: POSITIVE


# Classifying sentences using logistic regression technique...
# The result of the logistic classification is: NEGATIVE
# The result of the logistic classification is: POSITIVE
# The result of the logistic classification is: POSITIVE
# The result of the logistic classification is: NEGATIVE
# The result of the logistic classification is: POSITIVE
# The result of the logistic classification is: NEGATIVE
# The result of the logistic classification is: POSITIVE
# The result of the logistic classification is: NEGATIVE
# The result of the logistic classification is: NEGATIVE
# The result of the logistic classification is: POSITIVE
# The result of the logistic classification is: NEGATIVE
# The result of the logistic classification is: POSITIVE
# The result of the logistic classification is: NEGATIVE
# The result of the logistic classification is: NEGATIVE
# The result of the logistic classification is: POSITIVE
# The result of the logistic classification is: POSITIVE
# The result of the logistic classification is: POSITIVE
# The result of the logistic classification is: NEGATIVE
# The result of the logistic classification is: NEGATIVE
# The result of the logistic classification is: NEGATIVE
# The result of the logistic classification is: POSITIVE
# The result of the logistic classification is: POSITIVE
# The result of the logistic classification is: NEGATIVE
# The result of the logistic classification is: POSITIVE
# The result of the logistic classification is: POSITIVE
# The result of the logistic classification is: POSITIVE
# The result of the logistic classification is: NEGATIVE
# The result of the logistic classification is: NEGATIVE
# The result of the logistic classification is: NEGATIVE
# The result of the logistic classification is: POSITIVE
# The result of the logistic classification is: POSITIVE
# The result of the logistic classification is: POSITIVE
# The result of the logistic classification is: NEGATIVE
# The result of the logistic classification is: NEGATIVE
# The result of the logistic classification is: POSITIVE
# The result of the logistic classification is: POSITIVE
# The result of the logistic classification is: NEGATIVE
# The result of the logistic classification is: NEGATIVE
# The result of the logistic classification is: POSITIVE
# The result of the logistic classification is: POSITIVE
# The result of the logistic classification is: NEGATIVE
# The result of the logistic classification is: NEGATIVE
# The result of the logistic classification is: POSITIVE
# The result of the logistic classification is: POSITIVE
# The result of the logistic classification is: POSITIVE
# The result of the logistic classification is: NEGATIVE
# The result of the logistic classification is: POSITIVE
# The result of the logistic classification is: NEGATIVE
# The result of the logistic classification is: POSITIVE
# The result of the logistic classification is: POSITIVE