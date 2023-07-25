import datetime
import os

chat_log = []

def log(speaker, message):
    chat_log.append((speaker, message))

def log_and_print(speaker, message):
    print(f"{speaker}: {message}")
    log(speaker, message)

def save_chat_log():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Check if the chat_logs directory exists, if not create it
    if not os.path.exists('chat_logs'):
        os.makedirs('chat_logs')

    # Include the directory path in the filename
    with open(f'chat_logs/chat_log_{timestamp}.txt', 'w') as file:
        file.write(f"This is your chat log from the conversation from {timestamp}\n\n")
        for speaker, message in chat_log:
            file.write(f"{speaker}: {message}\n")
