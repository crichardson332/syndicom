import json
import csv
import pdb
from collections import defaultdict
from random import sample

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

if __name__ == "__main__":
    splits = ['dev']
    for sp in splits:
        filter_dataset(sp)