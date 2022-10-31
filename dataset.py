import json
import csv
import pdb
import util
import random
from pprint import pprint
from random import sample, choice
from atomic import node_iterator

def verify_event(event):
    if '___' in event:
        return False
    
    if 'PersonX' not in event:
        return False

    return True

def gen_dataset(split, num_samples=100):

    # open atomic file and parse by tabs
    filename = f'atomic/atomic2020_data-feb2021/{split}.tsv'
    last_event = ''
    events = []
    with open(filename) as f:
        rd = csv.reader(f, delimiter="\t", quotechar='"')
        for row in rd:
            event = row[0]
            if verify_event(event):
                if event != last_event:
                    events.append(event)
            last_event = event

    # write events out to file
    outfilename = 'output/events.txt'
    with open(outfilename, 'w') as f:
        for event in events:
            f.write(event)
            f.write('\n')
    
    # sample set
    sample_events = sample(events, num_samples)
    soutfilename = 'output/sample.txt'
    with open(soutfilename, 'w') as f:
        for event in sample_events:
            f.write(event)
            f.write('\n')

def gen_primitives(split, num_samples=30):

    # filter by desired relations
    relations = ['oReact', 'xIntent', 'oWant']
    filename = f'atomic/atomic2020_data-feb2021/{split}.tsv'

    # output file create/overwrite file
    outfile = f'sample_convo.jsonl'
    with open(outfile, 'w') as f:
        pass

    count = 0
    for subgraph in node_iterator(filename):
        oReact = [tr for tr in subgraph if 'oReact' in tr]
        xIntent = [tr for tr in subgraph if 'xIntent' in tr]
        oWant = [tr for tr in subgraph if 'oWant' in tr]
        if (not oReact) or (not xIntent) or (not oWant):
            continue

        # clean PersonX pronouns
        oReact = util.clean_pronouns(oReact, personX='I', personY='you')
        xIntent = util.clean_pronouns(xIntent, personX='I', personY='you')
        oWant = util.clean_pronouns(oWant, personX='I', personY='you')

        # get head node
        head = util.sanitize(util.to_past_tense(oReact[0][0]))
        if '___' in head:
            continue

        # build the pseudo-english convo
        pe_convo = [head]
        pe_convo.append(f'My reaction is: {random.choice(oReact)[2]}')
        pe_convo.append(f'My intent was: {random.choice(xIntent)[2]}')
        pe_convo.append(f'I wanted: {random.choice(oWant)[2]}')

        with open(outfile, 'a') as f:
            js = {"source": '[SEP]'.join(pe_convo)}
            json.dump(js, f)
            f.write('\n')

        count += 1
        if count >= num_samples:
            break

    # # write events out to file
    # outfilename = 'output/events.txt'
    # with open(outfilename, 'w') as f:
    #     for event in events:
    #         f.write(event)
    #         f.write('\n')
    
    # # sample set
    # sample_events = sample(events, num_samples)
    # soutfilename = 'output/sample.txt'
    # with open(soutfilename, 'w') as f:
    #     for event in sample_events:
    #         f.write(event)
    #         f.write('\n')

if __name__ == "__main__":
    splits = ['dev']
    for sp in splits:
        # gen_dataset(sp)
        gen_primitives(sp)