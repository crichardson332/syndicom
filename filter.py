import json
import csv
import pdb
from tqdm import tqdm
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
    
def parse_tails(split):
    infile = f'output/dataset/{split}.jsonl'
    with open(infile, 'r') as json_file:
        json_list = list(json_file)

    # open output file
    outfile = f'output/dataset/tails_{split}.jsonl'
    with open(outfile, 'w') as f:
        pass

    # write tails from dataset
    for json_str in tqdm(json_list):
        datum = json.loads(json_str)

        # strip out the tail
        for idx,statement in enumerate(datum['template']):
            if 'NEGATION' in statement:
                continue
            parts = statement.split(':')
            parts = [p for p in parts if p]
            tail = parts[-1]

            # removing leading whitespace
            if tail[0] == ' ':
                tail = tail[1:]

            # write tail to the file
            json_out = {'text': tail}
            with open(outfile, 'a') as f:
                json.dump(json_out, f)
                f.write('\n')




if __name__ == "__main__":
    splits = ['train']
    for sp in splits:
        # filter_dataset(sp)
        parse_tails(sp)