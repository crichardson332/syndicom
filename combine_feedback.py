import json
import csv
import util
from tqdm import tqdm
import pdb
from pprint import pprint
from pathlib import Path

def process_for_gpt_finetuning(sp):
    # final resting place
    outfile = f'output/dialogue_modeling/feedback/{sp}.jsonl'
    with open(outfile, 'r') as json_file:
        jlout = list(json_file)

    with open(outfile, 'w') as json_file:
        pass


    # get dialogues with feedback
    infile = f'output/feedback/{sp}.jsonl'
    with open(infile, 'r') as json_file:
        jlin = list(json_file)

    for jsin, jsout in zip(jlin,jlout):
        ref = json.loads(jsin)['feedback']
        try:
            datum = json.loads(jsout)
        except:
            pdb.set_trace()
        datum['feedback'] += ref

        # write new data
        with open(outfile, 'a') as json_file:
            json.dump(datum, json_file)
            json_file.write('\n')

if __name__ == "__main__":
    splits = ['dev']
    for sp in splits:
        process_for_gpt_finetuning(sp)