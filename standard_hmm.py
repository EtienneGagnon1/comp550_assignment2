import re
from nltk.tag import hmm
from nltk.probability import LaplaceProbDist, ConditionalFreqDist, FreqDist
from typing import List, Dict, AnyStr
from unicodedata import normalize
from nltk.tokenize import word_tokenize
from numpy import mean
import nltk
import argparse
from nltk.corpus import treebank


parser = argparse.ArgumentParser()

parser.add_argument('-laplace', help="adds laplace smoothing", action="store_true")
parser.add_argument('-lm', help="informs the character transitions in english using extra-text", action='store_true')
parser.add_argument('cipher', type=str)
args = parser.parse_args()


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


def find_accuracy_per_sentence(predicted_sentences: List, gold_standard_sentences: List):
    all_sentences_accuracy = []
    for i in range(len(predicted_sentences)):
        matching_characters = 0
        for i in range(len(predicted_sentences[i])):
            predicted_character = predicted_sentences[i]
            actual_character = gold_standard_sentences[i]
            if predicted_character == actual_character:
                matching_characters += 1
        accuracy = matching_characters/len(predicted_sentences[i])        
        all_sentences_accuracy.append(accuracy)
    return all_sentences_accuracy


def find_total_accuracy(predicted_sentences: List, gold_standard_sentences: List):
    matching_characters = 0
    for i in range(len(predicted_sentences)):
        for i in range(len(predicted_sentences[i])):
            predicted_character = predicted_sentences[i]
            actual_character = gold_standard_sentences[i]
            if predicted_character == actual_character:
                matching_characters += 1
    accuracy = matching_characters/sum(len(sentence) for sentence in predicted_sentences)
    return accuracy


def find_total_accuracy(predicted_sentences: List, gold_standard_sentences: List):
    predicted_string = ''.join(predicted_sentences)
    gold_standard_string = ''.join(gold_standard_sentences)

    matches = 0
    for i in range(len(predicted_string)):
        predicted_character = predicted_string[i]
        actual_character = gold_standard_string[i]

        if predicted_character == actual_character:
            matches += 1

    accuracy = matches/len(predicted_string)
    return accuracy


def extract_predicted_sequence(predicted_sequence: List) -> List:
    prediction = []
    for sentence in predicted_sequence:
        predicted_char_list = [prediction[1] for prediction in sentence]
        prediction.append(predicted_char_list)
    return prediction


def clean_additional_text(additional_text: str) -> str:
    def remove_unicode_tags(additional_text: str):
        unicode = re.compile(r'[^\x00-\x7F]+')
        remove_lineskip = re.compile("\n")

        additional_text = unicode.sub(r'[^\x00-\x7F]+', ' ', additional_text)
        additional_text = remove_lineskip.sub(' ', additional_text)

        return additional_text


    allowed_states = re.compile('[^a-z,.\s]')
    white_space_before_period = re.compile('(\s*\.)')

    additional_text = remove_unicode_tags(additional_text)

    lower_cased = additional_text.lower()
    allowed_states = allowed_states.sub('', lower_cased)
    allowed_states_nowhitespace = white_space_before_period.sub('.', allowed_states)

    return allowed_states_nowhitespace


def find_transition_frequency(additional_sentences: str):
    condtional_freq_dist = ConditionalFreqDist()
    unique_characters = set(additional_sentences)

    for character in unique_characters:
        matches = re.findall('(?<={})[a-z,.\s]'.format(character), additional_sentences)
        condtional_freq_dist[character] = FreqDist(matches)

    return condtional_freq_dist


def main():
    cipher_test, cipher_train, plaintext_test, plaintext_train = read_in_ciphers(args.cipher)

    training_set = []
    for i in range(len(cipher_train) - 1):
        training_units = turn_training_observation_into_nltk_format(cipher_train[i], plaintext_train[i])
        training_set.append(training_units)

    if args.lm:
        with open('frankenstein_ulysses_hrtofdarkness.txt', 'rb') as f:
            extra_text = f.read()
            extra_text = str(extra_text)
            normalize('NFKD', extra_text).encode('ascii')

        extra_text = clean_additional_text(extra_text)

        additional_text_transitions = find_transition_frequency(extra_text)
        hidden_markov_trainer = hmm.HiddenMarkovModelTagger(transitions=additional_text_transitions)
    else:
        hidden_markov_trainer = hmm.HiddenMarkovModelTagger()

    if args.laplace:
        tagger = hidden_markov_trainer.train(training_set, estimator=LaplaceProbDist)
    else:
        tagger = hidden_markov_trainer.train(training_set)

    test_set = turn_test_cipher_into_nltk_format(cipher_test)

    predictions = [tagger.tag(test_sentence) for test_sentence in test_set]
    predicted_sequence = extract_predicted_sequence(predictions)
    recomposed_sentences = [''.join(sentence) for sentence in predicted_sequence]

    print('\n')
    print('These sentences were decoded using the hidden markov model: \n')

    for sentence in recomposed_sentences:
        print(sentence)

    print('\n')

    sentence_accuracy = find_accuracy_per_sentence(recomposed_sentences, plaintext_test)

    for (counter, value) in enumerate(sentence_accuracy):
        print('The per token accuracy in sentence %i was %f' % (counter, value))

    print('\n')
    whole_text_accuracy = find_total_accuracy(recomposed_sentences, plaintext_test)
    print('The accuracy for the whole text was %s' % whole_text_accuracy)




