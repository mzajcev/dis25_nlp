import re
from .chat_logger import log, log_and_print, save_chat_log
from .preprocessing import clean_user_input
from .models import classify_sentence
from utils.create_synonyms import greetings_synonyms, exit_synonyms, analyze_synonyms

def generate_response(user_input, analyze_mode, confirm_analyze_mode):
    patterns = {
        r'(?i)({}).*'.format('|'.join(greetings_synonyms)): ("Hello and Welcome! How can I help you?", False, False, False),
        r'(?i).*(weather).*': ("I guess the weather is good. Nice, that you found the Easteregg. Please continue with a serious request.", False, False, False),
        r'(?i).*(help).*': ("For help and an explanation of my functions please read the according README file!", False, False, False),
        r'(?i).*(author|developer).*': ("This Chatbot is created and developed by Maurice Sielmann, Marc Pricken and Matthias Zajcev.", False, False, False),
        r'(?i).*({}).*'.format('|'.join(analyze_synonyms)): ("Great! You seem interested in analyzing the sentiment of a sentence. Confirm by typing 'y' or 'n' to cancel.", True, True, False),
        r'(?i).*({}).*'.format('|'.join(exit_synonyms)): ("You have terminated the program!", False, False, True)
    }

    for pattern, (response, new_analyze_mode, new_confirm_analyze_mode, exit_chat) in patterns.items():
        if re.match(pattern, user_input):
            log_and_print('Chatbot', response)
            return new_analyze_mode, new_confirm_analyze_mode, exit_chat
    
    log_and_print('Chatbot', "Sorry, I don't understand. Please try again or read the README file.")
    return analyze_mode, confirm_analyze_mode, False  # By default, it will not exit chat


def analyze_sentence():
    log_and_print('Chatbot', "Please type the sentence you want to analyze:")
    sentence_to_analyze = input('User: ')  # Wait for user input as the sentence to analyze
    log('User', sentence_to_analyze)
    cleaned_sentence = clean_user_input(sentence_to_analyze) #Clean the sentence here
    log_and_print('Chatbot', "Cleaned sentence - " + cleaned_sentence)

    technique = ask_for_technique()

    classify_sentence(cleaned_sentence, technique)  # Here I've put the Logistic Regression as the default model. Adjust it as per your needs.


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
        analyze_again = input("Would you like to analyze the sentiment of another sentence? (y/n): ")
        log('Chatbot', "Would you like to analyze the sentiment of another sentence? (y/n): ")
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