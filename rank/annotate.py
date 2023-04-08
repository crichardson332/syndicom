import json
import csv
import re
from pathlib import Path
import pdb
from pprint import pprint
from tabulate import tabulate, SEPARATING_LINE
import argparse
import random
import numpy as np
import os

def toggle_speaker(speaker):
    if speaker == 'A':
        return 'B'
    else:
        return 'A'

    
def main(split):

    outfile = f'output.jsonl'
    if not os.path.exists(outfile):
        with open(outfile, 'w') as f:
            pass

    with open(outfile, 'r') as f:
        try:
            last_datum = json.loads(f.readlines()[-1])
            start_index = last_datum['id'] + 1
        except IndexError:
            start_index = 0


    valids = ['123', '132', '213', '231', '312', '321']
    # parse input
    infile = f'{split}.jsonl'
    with open(infile, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list[start_index:]:
        datum = json.loads(json_str)
        if len(datum['response'].keys()) <= 1:
            continue

        responses = [
            {'text': datum['response']['valid'], 'source': 'valid'},
            {'text': datum['response']['gpt'], 'source': 'gpt'},
            {'text': datum['response']['gpt_finetuned'], 'source': 'gpt_finetuned'},
        ]
        random.shuffle(responses)
        dialogue = '\n'.join(datum['context'])

        # text wrapping
        for idx in range(len(responses)):
            responses[idx]['text'] = re.sub("(.{32})", "\\1\n", responses[idx]['text'], 0, re.DOTALL)

        print(tabulate([[dialogue] + [resp['text'] for resp in responses]], headers=['Context', 'Response 1', 'Response 2', 'Response 3'],tablefmt="simple_grid"))
        while True:
            inp = input('Order responses from best to worst: ')
            if any([v == inp.lower() for v in valids]):
                datum['ranked_responses'] = []
                for idx,exp in enumerate(responses):
                    exp['rank'] = inp.find(str(idx+1)) + 1
                    datum['ranked_responses'].append(exp)

                # write result
                with open(outfile, 'a') as f:
                    json.dump(datum, f)
                    f.write('\n')

                # onto next datum
                break
            print('Input MUST be some ordering of 123. Try again.')



if __name__ == '__main__':
    split = 'test'
    main(split)
