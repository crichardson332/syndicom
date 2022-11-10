import spacy
import pyinflect
import csv
import numpy as np
# from spacy.tokens import Token
from datamuse import Datamuse
import re
import pdb
from atomic import node_iterator
from tqdm import tqdm
from collections import defaultdict

api = Datamuse()
nlp = spacy.load('en_core_web_sm')
# initiate spacy and nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# Token.set_extension('pyinflect', default=False, force=True)


def sanitize(text):
    text = text.replace("I's", "my")
    text = text.replace("they's", "their")
    text = text.replace("you's", "your")
    text = text.replace("to he", "to him")
    text = text.replace("to she", "to her")
    text = text.replace("to they", "to them")
    text = text.replace("you was", "you were")
    return text

def to_past_tense(text):
    doc = nlp(sanitize(text))
    for i in range(len(doc)):
        token = doc[i]
        if token.tag_ in ['VBP', 'VBZ']:
            # print(token.text, token.lemma_, token.pos_, token.tag_) 
            try:
                text = text.replace(token.text, token._.inflect("VBD"))
            except TypeError:
                return text
    return text

def clean_pronouns(triplet_arr, personX='I', personY='you'):
    clean_arr = []
    for text_arr in triplet_arr:
        clean_text_arr = []
        for text in text_arr:
            text = re.sub('personx', 'PersonX', text, flags=re.IGNORECASE)
            text = re.sub('person x', 'PersonX', text, flags=re.IGNORECASE)
            text = re.sub(' x$', ' PersonX', text, flags=re.IGNORECASE)
            text = re.sub('^x ', 'PersonX ', text, flags=re.IGNORECASE)

            text = re.sub('persony', 'PersonY', text, flags=re.IGNORECASE)
            text = re.sub('person y', 'PersonY', text, flags=re.IGNORECASE)
            text = re.sub(' y$', ' PersonY', text, flags=re.IGNORECASE)
            text = re.sub('^y ', 'PersonY ', text, flags=re.IGNORECASE)

            text = text.replace('PersonX', personX)
            text = text.replace('PersonY', personY)
            clean_text_arr.append(text)
        clean_arr.append(clean_text_arr)

    return clean_arr

def jsonify_atomic(split, keep_blanks=False):
    filename = f'atomic/atomic2020_data-feb2021/{split}.tsv'
    head_map = defaultdict(lambda: defaultdict(list))

    # first get num_rows
    with open(filename, 'r') as f:
        rdr = csv.reader(f, delimiter="\t", quotechar='"')
        num_rows = sum(1 for row in rdr) 

    # have to ropen and re-parse the file
    with open(filename, 'r') as f:
        rdr = csv.reader(f, delimiter="\t", quotechar='"')

        print(f'Reading atomic {split} split...num_rows = {num_rows}')
        for triplet in tqdm(rdr, total=num_rows):
            head = triplet[0]
            relation = triplet[1]
            tail = triplet[2]

            # ignore empty triplets
            if triplet[2] == 'none':
                continue

            # ignore blanks unless told to keep
            if ('___' in head) or ('___' in tail):
                continue

            # just add to head dictionary
            head_map[head][relation].append(tail)

        print(f'Reading atomic {split} split...Done')

    return head_map


def opposite(word):
    opps = api.words(rel_ant=word)
    if len(opps) > 0:
        return api.words(rel_ant=word)[0]['word']
    else:
        return None
    
