import datetime
import os

# Create empty list to store chat logs
chat_log = []

# Function that takes the name and message and appends those to the chat_log list
def log(speaker, message):
    chat_log.append((speaker, message))

# Function that takes the name and message, prints them and calls the pre defined Function 'log'
def log_and_print(speaker, message):
    print(f"{speaker}: {message}") 
    log(speaker, message)

# Function saves the chat log into a text file in a directory named 'chat_logs'
def save_chat_log():
    # Create current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Check if directory 'chat_logs' exists or if not create it
    if not os.path.exists('chat_logs'):
        os.makedirs('chat_logs')

    # Open new txt file and include current timestamp in file name
    with open(f'chat_logs/chat_log_{timestamp}.txt', 'w') as file:
        file.write(f"This is your chat log from the conversation from {timestamp}\n\n")
        
        # Loop through each tuple in the chat_log list and write each name and message tuple into the file
        for speaker, message in chat_log:
            file.write(f"{speaker}: {message}\n")
