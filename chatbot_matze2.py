import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from spellchecker import SpellChecker
import joblib
import datetime

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
spell = SpellChecker()

loaded_clf_logistic = joblib.load('classifier.logistic_regression')
loaded_vectorizer_logistic = joblib.load('vectors.logistic_regression')
loaded_clf_naive = joblib.load('classifier.naive_bayes')
loaded_vectorizer_naive = joblib.load('vectors.naive_bayes')

list_syn = {
    'hello': ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening'],
    'analyze': ['analyze']
}

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
        r'(?i)({}).*'.format('|'.join(list_syn['hello'])): "How can I help you?",
        r'(?i)(quit|exit).*': "See You!",
        r'(?i)({}).*'.format('|'.join(list_syn['analyze'])): "Great! Please enter the first sentence you would like to analyze."
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
        generated_response = generate_response(user_input)
        if generated_response:
            if 'See You!' in generated_response:
                break
            continue

        cleaned_sentence = clean_user_input(user_input)
        log_and_print('Chatbot', 'Cleaned sentence - ' + cleaned_sentence)
        technique = input("Which technique would you like to use (Logistic Regression/Naive Bayes)? ")
        log('Chatbot', "Which technique would you like to use (Logistic Regression/Naive Bayes)? ")  # Log the question
        log('User', technique)
        classify_sentence(cleaned_sentence, technique.lower())
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