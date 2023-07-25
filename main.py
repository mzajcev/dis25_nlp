# Import relevant modules
from chatbot import interaction, models, preprocessing, chat_logger
from utils import nltkmodules

# Main function for the chatbot
def main():
    # Initialize variables for analysis mode, confirm analysis mode, exit chat, sentence to analyze, and technique
    analyze_mode = False
    confirm_analyze_mode = False
    exit_chat = False
    sentence_to_analyze = None
    technique = None

    # The chatbot introduces itself and instructs the user
    chat_logger.log_and_print('Chatbot', "Hello, I am Mr.C-Bot and I am here to help. Please make sure to read the README file before talking to me. What can I do for you?")

    # While the chatbot is not set to exit
    while not exit_chat:
        # Get input from the user
        user_input = input('User: ')
        # Log the user's input
        chat_logger.log('User', user_input)
        # Preprocess and correct the user's input
        corrected_user_input = preprocessing.clean_user_input(user_input)

        # If the chatbot is in analysis mode
        if analyze_mode:
            # If the chatbot is in confirm analysis mode
            if confirm_analyze_mode:
                # If the user confirms
                if user_input.lower() == 'y':
                    # Reset the confirm analysis mode for the next round
                    confirm_analyze_mode = False
                    # Analyze the sentence
                    interaction.analyze_sentence()
                    # While the user wants to analyze again
                    while interaction.ask_to_analyze_again():
                        # Analyze the sentence
                        interaction.analyze_sentence()
                    # Exit analysis mode
                    analyze_mode = False

                # If the user cancels
                elif user_input.lower() == 'n':
                    # Reset analysis mode for the next round
                    analyze_mode = False
                    # Also reset confirm analysis mode for the next round
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

    # Ask the user if they want to save the chat log
    interaction.ask_to_save_chat_log()


# If the script is run directly (not imported)
if __name__ == "__main__":
    # Run the main function
    main()
