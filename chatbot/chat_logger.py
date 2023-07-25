# Importing necessary modules
import datetime
import os

# Initialize an empty list to store chat logs
chat_log = []

# This function takes in a speaker's name and message, then appends them as a tuple to the chat_log list
def log(speaker, message):
    chat_log.append((speaker, message))

# This function takes in a speaker's name and message, prints them, and also calls the log function to append them to chat_log
def log_and_print(speaker, message):
    print(f"{speaker}: {message}")  # Print the speaker and message
    log(speaker, message)  # Call the log function to store the message in the chat log

# This function saves the chat log into a text file in a directory named 'chat_logs'
def save_chat_log():
    # Create a timestamp using the current time in the format "Year-Month-Day_Hour-Minute-Second"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Check if the directory 'chat_logs' exists, if not, create it
    if not os.path.exists('chat_logs'):
        os.makedirs('chat_logs')

    # Open a new file in write mode, the filename includes the timestamp
    with open(f'chat_logs/chat_log_{timestamp}.txt', 'w') as file:
        # Write an introduction line into the file
        file.write(f"This is your chat log from the conversation from {timestamp}\n\n")
        
        # Loop through each tuple in the chat_log list
        for speaker, message in chat_log:
            # Write each speaker's name and message into the file
            file.write(f"{speaker}: {message}\n")
