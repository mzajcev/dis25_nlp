import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from spellchecker import SpellChecker
import joblib
import datetime
from create_synonyms import greetings_synonyms, analyze_synonyms, exit_synonyms


lemmatizer = WordNetLemmatizer()
spell = SpellChecker()

stop_words_orig = set(stopwords.words('english'))
exclude_stopword = {'not'}
stop_words = ([word for word in stop_words_orig if word not in exclude_stopword])

loaded_clf_logistic = joblib.load('classifier.logistic_regression')
loaded_vectorizer_logistic = joblib.load('vectors.logistic_regression')
loaded_clf_naive = joblib.load('classifier.naive_bayes')
loaded_vectorizer_naive = joblib.load('vectors.naive_bayes')

chat_log = []

def log(speaker, message):
    chat_log.append((speaker, message))

def log_and_print(speaker, message):
    print(f"{speaker}: {message}")
    log(speaker, message)

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


def classify_sentence(sentence, technique):
    if technique == 'logistic':
        vector = loaded_vectorizer_logistic
        classifier = loaded_clf_logistic
    elif technique == 'naive':
        vector = loaded_vectorizer_naive
        classifier = loaded_clf_naive
    else:
        log_and_print("Chatbot", "Invalid technique. Please choose either 'logistic' or 'naive'.")
        return

    vectorized_sentence = vector.transform([sentence])
    classification_result = classifier.predict(vectorized_sentence)
    response = f"The result of the {technique} classification is: {classification_result[0]}"
    log_and_print('Chatbot', response)

def generate_response(user_input):
    patterns = {
        r'(?i)({}).*'.format('|'.join(greetings_synonyms)): "How can I help you?",
        r'(?i)({}).*'.format('|'.join(exit_synonyms)): "See You!",
        r'(?i)({}).*'.format('|'.join(analyze_synonyms)): "Great! Please enter the first sentence you would like to analyze."
    }

    for pattern, response in patterns.items():
        if re.match(pattern, user_input):
            log_and_print('Chatbot', response)
            return response
    

def save_chat_log():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f'chat_log_{timestamp}.txt', 'w') as file:
        file.write(f"This is your chatlog from the conversation from {timestamp}\n\n")
        for speaker, message in chat_log:
            file.write(f"{speaker}: {message}\n")

def main():
    log_and_print('Chatbot', "Hello, I am Mr.C-Bot and I am here to help. Please make sure to read the README file before talking to me. What can I do for you?")
    while True:
        user_input = input("User: ")
        log('User', user_input)

        # Apply spell check on user_input before further processing
        corrected_user_input = clean_user_input(user_input)

        generated_response = generate_response(corrected_user_input)
        if generated_response:
            if 'See You!' in generated_response:
                break
            continue

        cleaned_sentence = corrected_user_input  # We have already cleaned it, no need to clean again
        log_and_print('Chatbot', 'Cleaned sentence - ' + cleaned_sentence)
        while True:
            technique = input("Which technique would you like to use (logistic/naive)? ").lower()
            log('Chatbot', "Which technique would you like to use (logistic/naive)? ")  # Log the question
            log('User', technique)
            if 'logistic' in technique:
                technique = 'logistic'
                break
            elif 'naive' in technique:
                technique = 'naive'
                break
            else:
                log_and_print('Chatbot', "Invalid technique. Please include either 'logistic' or 'naive' in your response.")
                
        classify_sentence(cleaned_sentence, technique)
        cont = input("Do you want to analyze another sentence? (y/n): ")
        log('Chatbot', "Do you want to analyze another sentence? (y/n): ")  # Log the question
        log('User', cont)
        if cont.lower() == 'n':
            break

    log_choice = input("Would you like to save the chat log? (y/n): ")
    log('Chatbot', "Would you like to save the chat log? (y/n): ")  # Log the question
    log('User', log_choice)
    if log_choice.lower() == 'y':
        log_and_print('Chatbot', "Alright. Saving chat log... Good Bye!")
        save_chat_log()

    if log_choice.lower() == 'n':
        log_and_print('Chatbot', "Alright. Have a good one!")

if __name__ == "__main__":
    main()