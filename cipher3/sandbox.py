import nltk
from nltk.tag import hmm
from nltk.probability import LaplaceProbDist
from typing import List, Dict, AnyStr
from numpy import mean
import argparse
from nltk.corpus import treebank
import re


treebank_sentence = [' '. join(sentence) for sentence in treebank.sents()]

lower_cased = [sentence.lower() for sentence in treebank_sentence]
allowed_states = re.compile('[^a-z,.\s]')

print(allowed_states.('a', ''))
''.join([allowed_states.sub('', character) for character in lower_cased[0]])

for i in treebank_sentence[0]:
    print(type(i))




