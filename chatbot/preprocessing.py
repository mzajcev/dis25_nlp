import re
import nltk
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer
from string import punctuation
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from googletrans import Translator

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
