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

def gen_gpt(split, num_samples=20):

    # filter by desired relations
    relation_set = ['IsAfter', 'IsBefore', 'xNeed', 'xAttr', 'xEffect', 'xReact', 'xWant', 'xIntent']
    filename = f'atomic/atomic2020_data-feb2021/{split}.tsv'

    # output file create/overwrite file
    outfile = f'sample.jsonl'
    with open(outfile, 'w') as f:
        pass

    count = 0
    all_convos = []
    all_examples = []
    primitives = []
    for subgraph in tqdm(node_iterator(filename)):
        relations = random.choices(relation_set, k=3)
        relation_map_raw = {}
        for rel in relations:
            relation_map_raw[rel] = [tr for tr in subgraph if rel in tr]

        # continue if one of the relations is empty
        if not all(relation_map_raw.values()):
            continue

        # clean PersonX pronouns
        relation_map = relation_map_raw
        for key,val in relation_map.items():
            relation_map[key] = util.clean_pronouns(relation_map_raw[key], personX='I', personY='you')

        # get head node
        try: 
            head = util.sanitize(util.to_past_tense(relation_map[key][0][0]))
        except IndexError:
            pdb.set_trace()
        if '___' in head:
            continue

        # build the pseudo-english convo
        pe_convo = [head + '.']
        # pdb.set_trace()
        for key,val in relation_map.items():
            turn = random.choice(val)[2]
            pe_convo.append(f'({key}) {turn}')
        # pe_convo.append(f'')
        # pe_convo.append(f'{turn2}')
        # pe_convo.append(f'You must be: {turn3}')
        speaker = 'PersonY'
        example = ''
        # for turn in pe_convo:
        #     speaker = toggle_speaker(speaker)
        #     example = example + f'{speaker}: {turn} ' 

        # all_examples.append(example)
        all_convos.append(pe_convo)

        # track the original primitives
        prim = [subgraph[0][0]]
        for key,val in relation_map_raw.items():
            triplet = random.choice(val)
            prim.append(f'{prefix_map[triplet[1]]}{triplet[2]}.')
            # relation_map_raw
        primitives.append(prim)


        # with open(outfile, 'a') as f:
        #     js = {"source": '[SEP]'.join(pe_convo)}
        #     json.dump(js, f)
        #     f.write('\n')


        # count += 1
        # if count >= num_samples:
        #     break
    # output = random.choices(all_examples, k=num_samples)
    # output = random.choices(all_convos, k=num_samples)
    # for ex in output:
    #     print(f'Input:')
    #     print(f'   PersonX: {ex[0]}')
    #     print(f'   PersonY: {ex[1]}')
    #     print(f'   PersonX: {ex[2]}')
    #     print(f'   PersonY: {ex[3]}')
    #     print(f'Output:')
    #     print(f'   PersonX: ')
    #     print(f'   PersonY: ')
    #     print(f'   PersonX: ')
    #     print(f'   PersonY: ')
    output = random.choices(primitives, k=num_samples)
    for ex in output:
        if len(ex) != 4:
            pdb.set_trace()
            # continue
        print(f'Background:')
        print(f'   {ex[0]}')
        print(f'   {ex[1]}')
        print(f'   {ex[2]}')
        print(f'   {ex[3]}')
        print(f'Dialogue:')
        print(f'   PersonX: ')
        print(f'   PersonY: ')
        print(f'   PersonX: ')
        print(f'   PersonY: ')
    # pdb.set_trace()

def common_items(list1, list2):
    return list(set(list1) & set(list2))

def gen_gpt_inputs(split, num_samples=None):
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
    # splits = ['dev']
    splits = ['train']
    for sp in splits:
        # gen_dataset(sp)
        # gen_primitives(sp)
        # gen_intent_attr(sp)
        # gen_gpt(sp)
        # gen_mturk_sample()
        gen_gpt_inputs(sp, num_samples=10000)