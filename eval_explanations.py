
import json
import csv
from tqdm import tqdm
import pdb
from pprint import pprint
from pathlib import Path
import numpy as np


def eval_expl():

    infile = f'output/evaluation/output_dristi.jsonl'
    with open(infile, 'r') as json_file:
        json_list = list(json_file)


    # compute MRR for GPT-3
    mrr_tot = {
        'gpt': 0,
        'avg_human': 0,
        'worst_human': 0,
        'best_human': 0,
    }
    count = 0
    for json_str in json_list:
        datum = json.loads(json_str)

        # add 1 to every rank to normalize to 1, 2, 3...
        gpt_rank = [1 + ex['rank'] for ex in datum['explanations'] if ex['source'] == 'gpt-3'][0]
        mrr_tot['gpt'] += 1 / gpt_rank

        # human average, worst, and best
        human_ranks = [1 + ex['rank'] for ex in datum['explanations'] if ex['source'] == 'mturk']
        mrr_tot['avg_human'] += 1/(np.mean(human_ranks))
        mrr_tot['worst_human'] += 1/(np.max(human_ranks))
        mrr_tot['best_human'] += 1/(np.min(human_ranks))
        
        count += 1

    mrr = mrr_tot
    for key in mrr_tot.keys():
        mrr[key] = mrr_tot[key] / count
    pprint(mrr)
    pdb.set_trace()

if __name__ == "__main__":
    eval_expl()