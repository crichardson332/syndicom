import json
import csv
import pdb
from tqdm import tqdm
from collections import defaultdict
from random import sample
import random

random.seed(2022)

def filter_dataset(split):
    # open atomic file and parse by tabs
    filename = f'atomic2020_data-feb2021/{split}.tsv'
    last_event = ''
    events = []
    filtered_triplets = defaultdict(list)
    with open(filename) as f:
        rows = csv.reader(f, delimiter="\t", quotechar='"')
        for row in rows:
            head, relation, tail = row
            datum = {
                'head': head,
                'relation': relation,
                'tail': tail
            }
            filtered_triplets[relation].append(datum)

    # open separate file for each relation
    for key in filtered_triplets.keys():
        fname = f'data/{split}/{key}.jsonl'
        fobj = open(fname, 'w')

        # write all the triplets to this file
        for triplet in filtered_triplets[key]:
            json.dump(triplet, fobj)
            fobj.write('\n')
            # pdb.set_trace()

        # close all the files after truncating the newline
        fobj.truncate(fobj.tell()-1)
        fobj.close()

def gen_examples(split, num_examples=30):

    outfile = f'examples/gpt_examples.txt'
    with open(outfile, 'w') as f:
        pass

    filename = f'output/templates/{split}.jsonl'
    with open(filename, 'r') as json_file:
        json_list = list(json_file)
        random.shuffle(json_list)

    out_str = ''
    for i in range(num_examples):
        datum = json.loads(json_list[i])
        out_str += f'\n\ntemplate:'
        for turn in datum['template']:
            out_str += f'\n    {turn}'

        out_str += f'\ndialogue:'
        for j in range(len(datum['template'])):
            j_even = ((j % 2) == 0)
            speaker = 'X' if j_even else 'Y'
            out_str += f'\n    Person{speaker}:'

    with open(outfile, 'w') as f:
        f.write(out_str)

if __name__ == "__main__":
    splits = ['train']
    for sp in splits:
        # filter_dataset(sp)
        gen_examples(sp)