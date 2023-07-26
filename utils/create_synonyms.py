import re
import nltk
from nltk.corpus import wordnet

# List of words for greetings, analyze request and exit request with the goal to add synonyms to each list by using the 'generate_synonyms' function 
greetings_list = [
    'Hi', 'Hey', 'Greetings', 'Howdy', 'Salutations', 'Hiya', 'Yo', 'Hi there', "What's up", 'Hey there',
    'Hola', 'Bonjour', 'Ciao', 'Aloha', 'Wassup', "How's it going", "What's happening", "How's things", 'Good day',
    'Good morning', 'Good afternoon', 'Good evening', 'Sup', 'Hey buddy', 'How do you do', 'How are you', "How's life"
]

analyze_list = [
    'Examine', 'Investigate', 'Study', 'Scrutinize', 'Inspect', 'Evaluate', 'Assess', 'Review', 'Survey',
    'Probe', 'Scan', 'Monitor', 'Check', 'Diagnose', 'Interpret', 'Break down', 'Dissect', 'Explore',
    'Research', 'Test', 'Anatomize', 'Watch', 'Closely examine', 'Question', 'Delve into'
]

exit_list = [
    'Leave', 'Depart', 'Go out', 'Withdraw', 'Egress', 'Vacate', 'Evacuate', 'Abandon', 'Retreat', 'Quit',
    'Flee', 'Bail out', 'Escape', 'Scurry', 'Clear out', 'Walk out', 'Step out', 'Take off', 'End',
    'Pull out', 'Pass away', 'Decamp', 'Move out', 'Say goodbye', 'Bid farewell', 'Make an exit'
]

# Function to generate synonyms for each list
def generate_synonyms(word_list):
    list_syn = []
    
    for word in word_list:
        synonyms = []

        # Iterate through the set of synonyms that match the word from list
        for syn in wordnet.synsets(word):
            # Iterate through each synonym in the set of synonyms
            for lem in syn.lemmas():

                # Remove any special characters
                lem_name = re.sub('[^a-zA-Z0-9 \n\.]', ' ', lem.name())
                # Append each synonym to the list of synonyms and convert to lowercase
                synonyms.append(lem_name.lower())  

        # Add the word itself to the list of synonyms and convert to lowercase
        synonyms.append(word.lower())

        # Remove duplicates
        synonyms = list(set(synonyms))
        
        # Extend the list with the synonyms for the current word
        list_syn.extend(synonyms)

    # Removing duplicates from the combined list
    list_synonyms = list(set(list_syn))
    
    return list_synonyms

# Generate lists of synonyms for each word list
greetings_synonyms = generate_synonyms(greetings_list)
analyze_synonyms = generate_synonyms(analyze_list)
exit_synonyms = generate_synonyms(exit_list)
