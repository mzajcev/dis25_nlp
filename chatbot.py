import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
nltk.download('stopwords')
nltk.download('wordnet')
import re 
from spellchecker import SpellChecker
import string
from string import punctuation

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation
from spellchecker import SpellChecker
import re

# English Stopwords for clean_user_input function
stop_words = set(stopwords.words("english"))
# Spellchecker
spell = SpellChecker()
# Lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_user_input(user_input):
    # Remove HTML tags from user input
    user_input_no_html = re.sub('<.*?>', '', user_input)

    # Word Tokenization
    user_input_token = word_tokenize(user_input_no_html)

    # Lowercase and Remove punctuation
    user_input_punct_lower = [x.lower() for x in user_input_token if x not in punctuation]

    # Remove Numbers
    user_input_no_num = [x for x in user_input_punct_lower if not x.isdigit()]

    # Spellchecker
    correct_words = [spell.correction(word) for word in user_input_no_num]

    # Remove Stopwords
    filtered_words = [word for word in correct_words if word not in stop_words]

    # Lemmatization
    base_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    # Join words into sentence
    clean_response = ' '.join(base_words)

    return clean_response

def generate_response(user_input, whole_input):
    patterns = {
        r'test\s(.*)': "{}".format(user_input),  # Respond with the test input
        r'logistic$': "{}".format(whole_input),  # Respond with the whole input for logistic regression
        r'naive$': "{}".format(whole_input)      # Respond with the whole input for naive bayes
    }

    for pattern, response in patterns.items():
        match = re.match(pattern, user_input)
        if match:
            if pattern == r'logistic$' or pattern == r'naive$':
                return None  # For logistic and naive, don't generate any response
            else:
                return response  # Return the test input for 'test' command
    return "I'm sorry, but I'm not sure I understand."

def chat():
    print("Hi")
    whole_input = ""

    while True:
        user_input = input("You: ")

        response = generate_response(user_input, whole_input)
        if response is not None:
            print("Mr. C-Bot:", response)

        match_test = re.match(r'test\s(.*)', user_input)
        if match_test:
            whole_input = match_test.group(1)

        if user_input.lower() == 'logistic':
            print(whole_input)  # Print the whole input for logistic regression

        if user_input.lower() == 'naive':
            print(whole_input)  # Print the whole input for naive bayes

        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

# Example usage
chat()
