import re
import nltk
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer
from string import punctuation
from nltk.corpus import stopwords
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()
spell = SpellChecker()

stop_words_orig = set(stopwords.words('english'))
exclude_stopword = {'not', 'against', 'nor', 'no'}
stop_words = ([word for word in stop_words_orig if word not in exclude_stopword])

def clean_user_input(user_input):
    user_input_no_html = re.sub('<.*?>', '', user_input)
    user_input_token = word_tokenize(user_input_no_html)
    user_input_punct_lower = [x.lower() for x in user_input_token if x not in punctuation]
    user_input_no_num = [x for x in user_input_punct_lower if not x.isdigit()]
    correct_words = [spell.correction(word) for word in user_input_no_num]
    filtered_words = [word for word in correct_words if word not in stop_words and word is not None]
    base_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    clean_response = ' '.join(base_words)
    return clean_response