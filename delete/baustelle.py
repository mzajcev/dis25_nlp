import re
from ..utils.nltkmodules import *
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from spellchecker import SpellChecker
import joblib
import datetime
from ..utils.create_synonyms import greetings_synonyms, analyze_synonyms, exit_synonyms

lemmatizer = WordNetLemmatizer()
spell = SpellChecker()

stop_words_orig = set(stopwords.words('english'))
exclude_stopword = {'not', 'against', 'nor', 'no'}
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
    response = f"The result of the {technique} classification is: {classification_result[0]}"
    log_and_print('Chatbot', response)

def generate_response(user_input, analyze_mode, confirm_analyze_mode):
    patterns = {
        r'(?i)({}).*'.format('|'.join(greetings_synonyms)): ("How can I help you?", False, False, False),
        r'(?i)(weather).*': ("I guess the weather is good. Nice, that you found the Easteregg. Please continue with a serious request.", False, False, False),
        r'(?i)(help).*': ("For help and an explaination of my functions please read the according README file!", False, False, False),
        r'(?i)(author|developer).*': ("This Chatbot is created and developed by Maurice Sielmann, Marc Pricken and Matthias Zajcev.", False, False, False),
        r'(?i)({}).*'.format('|'.join(exit_synonyms)): ("You have terminated the program!", False, False, True),
    }

    for pattern, (response, new_analyze_mode, new_confirm_analyze_mode, exit_chat) in patterns.items():
        if re.match(pattern, user_input):
            log_and_print('Chatbot', response)
            return new_analyze_mode, new_confirm_analyze_mode, exit_chat
    
    if any(word in user_input.lower() for word in analyze_synonyms):
        log('User', user_input)
        log_and_print('Chatbot', "Great! You seem interested in analyzing a sentence. Confirm by typing 'y' or 'n' to cancel.")
        return True, True, False  # analyze_mode is True, confirm_analyze_mode is True
    log('User', user_input)
    log_and_print('Chatbot', "Sorry, I don't understand. Please try again or read the README file.")
    return analyze_mode, confirm_analyze_mode, False  # By default, it will not exit chat

def save_chat_log():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f'chat_log_{timestamp}.txt', 'w') as file:
        file.write(f"This is your chat log from the conversation from {timestamp}\n\n")
        for speaker, message in chat_log:
            file.write(f"{speaker}: {message}\n")

def analyze_sentence():
    log_and_print('Chatbot', "Please type the sentence you want to analyze:")
    sentence_to_analyze = input('User: ')  # Wait for user input as the sentence to analyze
    log('User', sentence_to_analyze)
    cleaned_sentence = clean_user_input(sentence_to_analyze) #Clean the sentence here
    log_and_print('Chatbot', "Cleaned sentence - " + cleaned_sentence)

    technique = ask_for_technique()

    classify_sentence(cleaned_sentence, technique)  # Here I've put the Logistic Regression as the default model. Adjust it as per your needs.

def main():
    analyze_mode = False
    confirm_analyze_mode = False
    exit_chat = False
    sentence_to_analyze = None
    technique = None

    log_and_print('Chatbot', "Hello, I am Mr.C-Bot and I am here to help. Please make sure to read the README file before talking to me. What can I do for you?")

    while not exit_chat:
        user_input = input('User: ')
        log('User', user_input)
        corrected_user_input = clean_user_input(user_input)  # Directly assign user's input without correction

        if analyze_mode:
            if confirm_analyze_mode:
                if user_input.lower() == 'y':
                    confirm_analyze_mode = False  # Reset for the next round

                    analyze_sentence()
                    while ask_to_analyze_again():
                        analyze_sentence()
                        
                    analyze_mode = False

                elif user_input.lower() == 'n':
                    analyze_mode = False  # Also reset analyze_mode for the next round
                    confirm_analyze_mode = False  # Also reset confirm_analyze_mode for the next round
                    log_and_print('Chatbot', "Alright. Cancelling analysis. What would you like to do?")
                else:
                    log_and_print('Chatbot', "Invalid response. Please type 'y' to confirm or 'n' to cancel.")
            else:
                analyze_mode, confirm_analyze_mode, exit_chat = generate_response(corrected_user_input, analyze_mode, confirm_analyze_mode)
        else:
            analyze_mode, confirm_analyze_mode, exit_chat = generate_response(corrected_user_input, analyze_mode, confirm_analyze_mode)

    ask_to_save_chat_log()

def ask_for_technique():
    while True:
        technique = input("Which technique would you like to use (Logistic Regression/Naive Bayes)? ")
        log('Chatbot', "Which technique would you like to use (Logistic Regression/Naive Bayes)? ")
        log('User', technique)

        if 'logistic' in technique.lower():
            return 'Logistic Regression'
        elif 'naive' in technique.lower():
            return 'Naive Bayes'
        else:
            log_and_print('Chatbot', "Invalid input. Please choose either 'Logistic Regression' or 'Naive Bayes'.")

def ask_to_analyze_again():
    while True:
        analyze_again = input("Would you like to analyze another sentence? (y/n): ")
        log('Chatbot', "Would you like to analyze another sentence? (y/n): ")
        log('User', analyze_again)

        if analyze_again.lower() == 'y':
            return True
        elif analyze_again.lower() == 'n':
            ask_to_save_chat_log()
            return False
        else:
            log_and_print('Chatbot', "Invalid input. Please type 'y' to confirm or 'n' to cancel.")

def ask_to_save_chat_log():
    while True:
        log_choice = input("Would you like to save the chat log? (y/n): ")
        log('Chatbot', "Would you like to save the chat log? (y/n): ")
        log('User', log_choice)

        if log_choice.lower() == 'y':
            log_and_print('Chatbot', "Alright. Saving chat log... Good Bye! See you soon!")
            save_chat_log()
            exit()
        elif log_choice.lower() == 'n':
            log_and_print('Chatbot', "Alright. Have a good one! See you next time!")
            exit()
        else:
            log_and_print('Chatbot',"Invalid input! Only 'y' or 'n' allowed.")

if __name__ == "__main__":
    main()
