# README

## Chatbot: A Natural Language Processing (NLP) Based Chatbot

### Introduction

Welcome to our Chatbot! It is a conversational agent powered by natural language processing techniques. Developed by Maurice Sielmann, Marc Pricken and Matthias Zajcev, the bot's primary purpose is to analyze the sentiment of movie reviews and classify them according to predefined categories, using either Logistic Regression or Naive Bayes classifier. 

The chatbot performs the following key operations:

- Greets users and listen to their inputs.
- Cleans the input data by removing HTML tags, punctuations, digits, and stop words; correcting spellings, and lemmatizing the words.
- Classifys movie ratings based on user-selected technique: Logistic Regression or Naive Bayes and recommends movies based on the sentiment of the user-input.
- Keeps a record of the entire conversation that can be saved upon request.

### Dependencies
Please make sure to create a virtual enviroment and after that install the 'requirements.txt' before you try to use the bot to download the dependencies.

### Start
Start the bot by typing 'python .\main.py' into your console and enjoy!

### Beta Features

As this bot was intended to work in English language only, we decided to implement a translator based on google translate into this bot. You can try to write your input in German, or any other language. 
ENJOY! 

### Usage

1. **Initiate a conversation** - When you run the chatbot, it greets you and asks how it can help. 

2. **Request for sentence analysis** - To analyze a sentence, use keywords like "analyze" or "classify". The chatbot will recognize your intent and ask for your confirmation.

3. **Confirm analysis** - Type 'y' to confirm that you want to analyze a sentence, or 'n' to cancel the operation.

4. **Input a movie review for analysis** - If you confirmed in the previous step, the bot will ask you to type the sentence you want to analyze.

5. **Choose analysis technique** - After inputting the sentence, the bot will ask which technique you want to use for analysis: Logistic Regression or Naive Bayes. You can easily type 'logistic' or 'naive' to make a choice. Type in your choice.

6. **Receive classification result** - After you have selected the technique, the bot will analyze the sentence and output the sentiment result. Also you will receive a recommendation based on your sentiment result.

7. **Continue or end the conversation** - After one round of analysis, you can choose to analyze another sentence or end the conversation. The bot will ask for your decision.

8. **Save chat log** - Upon ending the conversation, the bot will ask if you would like to save the chat log. If you type 'y', it will save the conversation in a text file with a timestamp. If you type 'n', it will simply say goodbye and close the program.


### Additional Commands

- **Ask for help** - If you need help, you can use the keyword "help". The bot will basically send you here, to read the README.

- **Inquire about the author/developer** - If you want to know who created the bot, use keywords like "author" or "developer". The bot will provide the names of the developers.

- **Easteregg** - If you want to try our Easteregg ask the bot something about the weather with the keyword "weather" in your input ;-)

- **Exit the program** - If you want to exit the program, you can use keywords like "exit", "quit", "stop", "end" etc. The bot will terminate the program.

### Important Notes

- This chatbot uses logistic regression and naive bayes classifiers that are pre-trained on a specific dataset (IMDB). The categories into which the bot can classify sentences depend on the dataset it was trained on.
- The chatbot's understanding of user intent is based on keyword matching, so please use appropriate and logically keywords for your requests. Even if the bot can correct wrong entries, please write sensibly.


