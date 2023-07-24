from chatbot import interaction, models, preprocessing, chat_logger

def main():
    analyze_mode = False
    confirm_analyze_mode = False
    exit_chat = False
    sentence_to_analyze = None
    technique = None

    chat_logger.log_and_print('Chatbot', "Hello, I am Mr.C-Bot and I am here to help. Please make sure to read the README file before talking to me. What can I do for you?")

    while not exit_chat:
        user_input = input('User: ')
        chat_logger.log('User', user_input)
        corrected_user_input = preprocessing.clean_user_input(user_input)  # Directly assign user's input without correction

        if analyze_mode:
            if confirm_analyze_mode:
                if user_input.lower() == 'y':
                    confirm_analyze_mode = False  # Reset for the next round

                    interaction.analyze_sentence()
                    while interaction.ask_to_analyze_again():
                        interaction.analyze_sentence()
                        
                    analyze_mode = False

                elif user_input.lower() == 'n':
                    analyze_mode = False  # Also reset analyze_mode for the next round
                    confirm_analyze_mode = False  # Also reset confirm_analyze_mode for the next round
                    chat_logger.log_and_print('Chatbot', "Alright. Cancelling analysis. What would you like to do?")
                else:
                    chat_logger.log_and_print('Chatbot', "Invalid response. Please type 'y' to confirm or 'n' to cancel.")
            else:
                analyze_mode, confirm_analyze_mode, exit_chat = interaction.generate_response(corrected_user_input, analyze_mode, confirm_analyze_mode)
        else:
            analyze_mode, confirm_analyze_mode, exit_chat = interaction.generate_response(corrected_user_input, analyze_mode, confirm_analyze_mode)

    interaction.ask_to_save_chat_log()


if __name__ == "__main__":
    main()
