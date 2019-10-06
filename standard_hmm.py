from nltk.tag import hmm
from nltk.tokenize import word_tokenize
from nltk.probability import LaplaceProbDist
from typing import List, Dict, AnyStr
from numpy import mean
import sys


def read_in_ciphers(directory):
    def clean_strings_format(sample: List) -> List:
        cleaned_sample = [string.decode('unicode_escape') for string in sample]
        return cleaned_sample

    with open(directory + '/test_cipher.txt', 'rb') as f:
        test_cipher = f.readlines()
        test_cipher = clean_strings_format(test_cipher)

    with open(directory +'/train_cipher.txt', 'rb') as f:
        train_cipher = f.readlines()
        train_cipher = clean_strings_format(train_cipher)

    with open(directory + '/test_plain.txt', 'rb') as f:
        test_plaintext = f.readlines()
        test_plaintext = clean_strings_format(test_plaintext)

    with open(directory + '/train_plain.txt', 'rb') as f:
        train_cipher_plaintext = f.readlines()
        train_cipher_plaintext = clean_strings_format(train_cipher_plaintext)

    return test_cipher, train_cipher, test_plaintext, train_cipher_plaintext


def turn_training_observation_into_nltk_format(cipher: str, plaintext: str) -> List:
    def check_length_input(cipher=cipher, plaintext=plaintext):
        if len(cipher) != len(plaintext):
            raise IndexError('cipher and plaintext do not have matching length')
    check_length_input()

    cipher_characters = list(cipher)
    plaintext_characters = list(plaintext)

    nltk_input_format = []
    for i in range(len(cipher_characters)):
        character_pair = (cipher_characters[i], plaintext_characters[i])
        nltk_input_format.append(character_pair)
    return nltk_input_format


def turn_test_cipher_into_nltk_format(test_cipher):
    characterized = [list(cipher_sentence) for cipher_sentence in test_cipher]
    return characterized


def tag_test_set(test_cipher: List) -> List:
    corrected_sentences = []
    for sentence in test_cipher:
        character_sentence = list(sentence)
        tagger_output = tagger.tag(character_sentence)
        tagged_sentence = ''.join(tagger_output)
        corrected_sentences.append(tagged_sentence)

    return corrected_sentences


def find_per_token_accuracy(corrected_test_sentence: str, gold_standard: str) -> List:
    tokenized_corrected_sentence = word_tokenize(corrected_test_sentence)
    tokenized_gold_standard_sentence = word_tokenize(gold_standard)

    number_of_tokens = len(tokenized_gold_standard_sentence)

    accuracy_per_token = []
    for i in range(number_of_tokens):
        predicted_token = tokenized_corrected_sentence[i]
        actual_token = tokenized_gold_standard_sentence[i]

        token_length = len(actual_token)
        matches = 0
        if predicted_token == actual_token:
            matches = token_length
        else:
            for i in len(predicted_token):
                predicted_letter = predicted_token[i]
                actual_letter = actual_token[i]
                if predicted_letter == actual_letter:
                    matches += 1

        token_accuracy = matches/token_length
        accuracy_per_token.append(token_accuracy)

    return accuracy_per_token


def find_test_set_accuracy(predicted_sentences: List, gold_standard_sentences: List):
    sentence_accuracy = []
    for i in range(len(predicted_sentences)):
        accuracy = mean(find_per_token_accuracy(predicted_sentences[i], gold_standard_sentences[i]))
        sentence_accuracy.append(accuracy)
    return mean(sentence_accuracy)


def extract_predicted_sequence(predicted_sequence: List) -> List:
    prediction = []
    for sentence in predicted_sequence:
        predicted_char_list = [prediction[1] for prediction in sentence]
        prediction.append(predicted_char_list)
    return prediction


def answer_first_question(cipher: str):
    cipher_test, cipher_train, plaintext_test, plaintext_train = read_in_ciphers(cipher)

    training_set = []
    for i in range(len(cipher_train) - 1):
        print(i)
        training_units = turn_training_observation_into_nltk_format(cipher_train[i], plaintext_train[i])
        training_set.append(training_units)

    if "laplace" in sys.argv:
        hidden_markov_trainer = hmm.HiddenMarkovModelTrainer(estimator=LaplaceProbDist)
    else:
        hidden_markov_trainer = hmm.HiddenMarkovModelTrainer()

    tagger = hidden_markov_trainer.train_supervised(training_set)

    test_set = turn_test_cipher_into_nltk_format(cipher_test)

    predictions = [tagger.tag(test_sentence) for test_sentence in test_set]
    predicted_sequence = extract_predicted_sequence(predictions)
    recomposed_sentences = [''.join(sentence) for sentence in predicted_sequence]

    for sentence in recomposed_sentences:
        print(sentence)


def main():
    cipher_test, cipher_train, plaintext_test, plaintext_train = read_in_ciphers('cipher2')

    training_set = []
    for i in range(len(cipher_train) - 1):
        print(i)
        training_units = turn_training_observation_into_nltk_format(cipher_train[i], plaintext_train[i])
        training_set.append(training_units)

    hidden_markov_trainer = hmm.HiddenMarkovModelTrainer()
    smooth_tagger = hidden_markov_trainer.train_supervised(training_set, estimator=LaplaceProbDist)
    tagger = hidden_markov_trainer.train_supervised(training_set)


    test_set = turn_test_cipher_into_nltk_format(cipher_test)
    tagger.tag(test_set[0])
    smooth_tagger.tag(test_set[0])

    predictions = [smooth_tagger.tag(test_sentence) for test_sentence in test_set]
    predicted_sequence = extract_predicted_sequence(predictions)


    plaintext_test


