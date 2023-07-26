from chatbot import interaction, models, preprocessing, chat_logger
from utils import nltkmodules

def main():
    # Initialize variables for analysis mode, confirm analysis mode, exit chat, sentence to analyze, and technique
    analyze_mode = False
    confirm_analyze_mode = False
    exit_chat = False
    sentence_to_analyze = None
    technique = None

    # Give User Instructions
    chat_logger.log_and_print('Chatbot', "Hello, I am Mr.C-Bot and I am here to help. Please make sure to read the README file before talking to me. What can I do for you?")

    # While chatbot is not set to exit
    while not exit_chat:
        # Get User input and log it
        user_input = input('User: ')
        chat_logger.log('User', user_input)
        # Clean user input by calling function 'clean_user_input'
        corrected_user_input = preprocessing.clean_user_input(user_input)

        # If user input leads to analyze
        if analyze_mode:
            if confirm_analyze_mode:
                # If user confirms analyze mode
                if user_input.lower() == 'y':
                    # Reset confirm analysis mode for the next round
                    confirm_analyze_mode = False
                    # Analyze sentence
                    interaction.analyze_sentence()
                    # While the user wants to analyze again
                    while interaction.ask_to_analyze_again():
                        # Analyze the sentence
                        interaction.analyze_sentence()
                    # Exit analysis mode
                    analyze_mode = False

                # If user canacles analyze mode
                elif user_input.lower() == 'n':
                    # Reset analysis mode and confirm analysis mode for the next round
                    analyze_mode = False
                    confirm_analyze_mode = False
                    # Inform the user that the analysis has been cancelled
                    chat_logger.log_and_print('Chatbot', "Alright. Cancelling analysis. What would you like to do?")
                else:
                    # If the user inputs an invalid response
                    chat_logger.log_and_print('Chatbot', "Invalid response. Please type 'y' to confirm or 'n' to cancel.")
            else:
                # Generate a response based on the user's input
                analyze_mode, confirm_analyze_mode, exit_chat = interaction.generate_response(corrected_user_input, analyze_mode, confirm_analyze_mode)
        else:
            # Generate a response based on the user's input
            analyze_mode, confirm_analyze_mode, exit_chat = interaction.generate_response(corrected_user_input, analyze_mode, confirm_analyze_mode)

    # Ask user for chat log save
    interaction.ask_to_save_chat_log()


if __name__ == "__main__":
    main()
