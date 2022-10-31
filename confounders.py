import json
import csv
import pdb
from collections import defaultdict
from random import sample
from tkinter import W
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
import pyinflect
from pprint import pprint
from tqdm import tqdm
import random
from util import sanitize, to_past_tense, opposite

people_choices = [
    # 'you', 'he', 'she', 'Mark', 'James', 'Evan', 'Sarah', 'my mom', 'your dad'
    'you', 'he', 'she', 'they'
]

def HinderedBy(split):
    with open(f'data/{split}/HinderedBy.jsonl', 'r') as json_file:
        json_list = list(json_file)


    # create/overwrite file
    outfile = f'output/confounders/{split}/HinderedBy.jsonl'
    with open(outfile, 'w') as f:
        pass

    for i,json_str in enumerate(tqdm(json_list)):
        json_str = json_str.replace('Person X', 'PersonX')
        json_str = json_str.replace('Person Y', 'PersonY')
        datum = json.loads(json_str)
        tail = datum['tail']
        head = datum['head']
        # print(f'Head: {datum["head"]}')
        # print(f'Relation: {datum["relation"]}')
        # print(f'Tail: {datum["tail"]}')

        # replace generic pronouns
        tail = tail.replace('PersonX', 'Becky')
        head = head.replace('PersonX', 'she')
        tail = tail.replace('PersonY', 'James')
        head = head.replace('PersonY', 'him')

        # sanitize errors
        tail = sanitize(tail)
        head = sanitize(head)

        head_pt = to_past_tense(head)
        tail_pt = to_past_tense(tail)
        
        tail = sanitize(tail)
        head = sanitize(head)
        output = {
            'dialogue': {
                'speaker 1': tail_pt,
                'speaker 2': head_pt
            },
            'primitives': {
                'head': datum['head'],
                'relation': datum['relation'],
                'tail': datum['tail']
            }
        }

        # dump to file
        with open(outfile, 'a') as f:
            json.dump(output, f)
            f.write('\n')

def IsBefore(split):
    with open(f'data/{split}/IsBefore.jsonl', 'r') as json_file:
        json_list = list(json_file)

    # create/overwrite file
    outfile = f'output/confounders/{split}/IsBefore.jsonl'
    with open(outfile, 'w') as f:
        pass

    for i,json_str in enumerate(tqdm(json_list)):
        json_str = json_str.replace('Person X', 'PersonX')
        json_str = json_str.replace('Person Y', 'PersonY')
        datum = json.loads(json_str)
        tail = datum['tail']
        head = datum['head']

        # if head == "PersonX ruffles PersonY's feathers":
        #     pdb.set_trace()

        # replace generic pronouns
        tail = tail.replace('PersonX', 'I')
        head = head.replace('PersonX', 'you')
        tail = tail.replace('PersonY', random.choice(people_choices))
        head = head.replace('PersonY', 'they')

        tail = sanitize(tail)
        head = sanitize(head)

        head = to_past_tense(head)
        tail = to_past_tense(tail)

        # sanitize errors
        tail = sanitize(tail)
        head = sanitize(head)
        output = {
            'dialogue': {
                'speaker 1': tail,
                'speaker 2': f'And then {head}?'
            },
            'primitives': {
                'head': datum['head'],
                'relation': datum['relation'],
                'tail': datum['tail']
            }
        }

        # dump to file
        with open(outfile, 'a') as f:
            json.dump(output, f)
            f.write('\n')

def xAttr(split):
    with open(f'data/{split}/xAttr.jsonl', 'r') as json_file:
        json_list = list(json_file)

    # create/overwrite file
    outfile = f'output/confounders/{split}/xAttr.jsonl'
    with open(outfile, 'w') as f:
        pass

    for i,json_str in enumerate(tqdm(json_list)):
        json_str = json_str.replace('Person X', 'PersonX')
        json_str = json_str.replace('Person Y', 'PersonY')
        datum = json.loads(json_str)
        tail = datum['tail']
        head = datum['head']

        # if head == "PersonX ruffles PersonY's feathers":
        #     pdb.set_trace()

        # replace generic pronouns
        head = head.replace('PersonX', 'He')
        head = head.replace('PersonY', random.choice(people_choices))

        tail = opposite(tail)
        if not tail:
            # if no opposite is found, skip this example
            continue
        head = sanitize(head)

        head = to_past_tense(head)
        tail = to_past_tense(tail)

        # sanitize errors
        tail = sanitize(tail)
        head = sanitize(head)
        output = {
            'dialogue': {
                'speaker 1': head,
                'speaker 2': f'He must be {tail} then'
            },
            'primitives': {
                'head': datum['head'],
                'relation': datum['relation'],
                'tail': datum['tail']
            }
        }

        # dump to file
        with open(outfile, 'a') as f:
            json.dump(output, f)
            f.write('\n')


if __name__ == "__main__":
    splits = ['dev']
    for sp in splits:
        HinderedBy(sp)
        # IsBefore(sp)
        # xAttr(sp)