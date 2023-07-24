from baustelle import main, clean_user_input, classify_sentence
import re

def load_sentences_from_file(file_path):
    with open('test_data.csv', 'r') as file:
        sentences = file.read().splitlines()
    return sentences

def process_sentences(sentences, technique):
    for sentence in sentences:
        cleaned_sentence = clean_user_input(sentence)
        classify_sentence(cleaned_sentence, technique)

def main():
    file_path = "test_data.csv"  # replace with the path to your text file
    sentences = load_sentences_from_file(file_path)

    # Classify sentences using naive bayes technique
    print("\nClassifying sentences using naive bayes technique...")
    process_sentences(sentences, 'naive')

    # Classify sentences using logistic regression technique
    print("\nClassifying sentences using logistic regression technique...")
    process_sentences(sentences, 'logistic')

if __name__ == "__main__":
    main()
