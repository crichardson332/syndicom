import json
import csv
from tqdm import tqdm
import pdb
from pprint import pprint
from pathlib import Path
import numpy as np

def csv_to_dict(file_path):
    result_dict = {}

    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for key, value in row.items():
                try:
                    value = json.loads(value)
                except:
                    pass
                if key not in result_dict:
                    result_dict[key] = []
                result_dict[key].append(value)

    return result_dict


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

def eval_fb_correction(split, turns=[3]):

    hits = 0
    tot = 0
    for n_turns in turns:
        infile = f'output/mturk/response_selection/response/{split}{n_turns}.csv'
        results = csv_to_dict(infile)
        for ans in results['Answer.taskAnswers']:
            tot += 1
            if ans[0]['source']['syndicom']:
                hits += 1
    print(f'Accuracy: {hits/tot}')
    pdb.set_trace()


if __name__ == "__main__":
    # eval_expl()
    # turns = [3, 4, 5, 6]
    turns = [3, 4, 5]
    eval_fb_correction('test', turns)