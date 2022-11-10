import json
import csv
import pdb
import util
import random
from tqdm import tqdm
from pprint import pprint
from random import sample, choice
from atomic import node_iterator
import numpy as np

random.seed(2021)


# write prompts for each relation
prefix_map = {
    'IsAfter': 'Before this: ',
    'IsBefore': 'After this: ',
    'xNeed': 'Before that, PersonX needed: ',
    'xAttr': 'PersonX is seen as: ',
    'xEffect': 'As a result, PersonX will: ',
    'xReact': 'As a result, Personx feels: ',
    'xWant': 'As a result, PersonX wants: ',
    'xIntent' : 'PersonX wanted: ',
    'HinderedBy': 'This can be hindered by: ',
    'HasSubEvent': 'This includes the event/action: ',
}

def toggle_speaker(speaker):
    if speaker == 'PersonX':
        return 'PersonY'
    else:
        return 'PersonX'

def verify_event(event):
    if '___' in event:
        return False
    
    if 'PersonX' not in event:
        return False

    return True

def common_items(list1, list2):
    return list(set(list1) & set(list2))

def gen_dataset(split, num_samples=None):
    valid_relations = ['IsAfter', 'HasSubEvent', 'IsBefore', 'HinderedBy', 'xNeed', 'xAttr', 'xEffect', 'xReact', 'xWant', 'xIntent']
    min_turns = 3

    # output file create/overwrite file
    outfile = f'output/gpt/{split}.jsonl'
    with open(outfile, 'w') as f:
        pass

    atomic_json = util.jsonify_atomic(split)

    keys = list(atomic_json.keys())
    random.shuffle(keys)
    valid_keys = [key for key in keys if len(common_items(atomic_json[key].keys(), valid_relations)) >= (min_turns-1)]
    # get random subset
    if num_samples:
        subgraphs = random.choices([*atomic_json.items()],k=num_samples)
    else:
        subgraphs = atomic_json.items()

    # now loop over subgraphs. Each one will yield one dialogue
    print(f'Generating dataset...')
    for idx,subgraph in enumerate(tqdm(subgraphs)):
        head = subgraph[0]
        datum = {'template': [head], 'id': idx}

        # randomize number of turns
        num_turns = random.randint(min_turns,10)

        # find common set between the valid relations and relations in this subgraph
        # exit if this is not at least 3 (for 4 turns total)
        candidate_relations = common_items(subgraph[1].keys(), valid_relations)
        if len(candidate_relations) < min_turns:
            continue

        # reset num_turns if we have fewer valid relations to choose from
        num_turns = min(num_turns, len(candidate_relations)+1)

        # randomly choose relations
        relations = random.choices(candidate_relations, k=(num_turns-1))

        # for each relation, randomly choose a tail inference
        # for rel,tails in subgraph[1].items():
        for rel in relations:
            tails = subgraph[1][rel]

            # clean_tails = util.clean_pronouns([tails], personX='the person', personY='She')[0]
            tail = random.choice(tails)

            # append this to the template
            datum['template'].append(f'{prefix_map[rel]}{tail}.')

        with open(outfile, 'a') as f:
            json.dump(datum, f)
            f.write('\n')

    print(f'Generating dataset...Done')

if __name__ == "__main__":
    # splits = ['train', 'dev', 'test']
    splits = ['train']
    for sp in splits:
        gen_dataset(sp)