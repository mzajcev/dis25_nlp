import re
import nltk
from nltk.corpus import wordnet


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


# Function to generate synonyms for a given list of words
def generate_synonyms(word_list):
    list_syn = []
    
    for word in word_list:
        synonyms = []

        # Create synonyms for each word
        for syn in wordnet.synsets(word):
            for lem in syn.lemmas():

                # Remove any special characters from synonym strings
                lem_name = re.sub('[^a-zA-Z0-9 \n\.]', ' ', lem.name())
                synonyms.append(lem_name.lower())  # Convert to lowercase to make the list case-insensitive

        # Add the word itself to the list of synonyms
        synonyms.append(word.lower())

        # Removing duplicates for each word by converting to set and back to list
        synonyms = list(set(synonyms))
        
        # Extend the list of synonyms with the synonyms for the current word
        list_syn.extend(synonyms)

    # Removing duplicates from the combined list by converting to set and back to list
    list_synonyms = list(set(list_syn))
    
    return list_synonyms

# Generating synonym lists for each word list
greetings_synonyms = generate_synonyms(greetings_list)
analyze_synonyms = generate_synonyms(analyze_list)
exit_synonyms = generate_synonyms(exit_list)