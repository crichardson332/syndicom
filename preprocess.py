import json
import csv
from tqdm import tqdm
import pdb
from pprint import pprint
from pathlib import Path

def preprocess_binary_classification(split):
    infile = f'output/dataset/{split}.jsonl'
    with open(infile, 'r') as json_file:
        json_list = list(json_file)

    outfile = f'output/huggingface/syndicom/{split}.jsonl'
    for json_str in json_list:
        datum = json.loads(json_str)
        context = '\n'.join(datum['dialogue'][0:-1])
        valid = context + '[SEP]' + datum['dialogue'][-1]
        invalid = context + '[SEP]' + datum['negations'][-1]
        dout1 = {
            'text': valid,
            'label': 1
        }
        dout2 = {
            'text': invalid,
            'label': 0
        }

        with open(outfile, 'a') as f:
            json.dump(f, dout1)
            f.write('\n')
            json.dump(f, dout2)
            f.write('\n')


def add_explanations(split):
    # output file create/overwrite file
    outfile = f'output/final/{split}.jsonl'

    # initialize final data dict
    data = {}

    # get dialogues
    infile_dlg = f'output/dataset/{split}.jsonl'
    with open(infile_dlg, 'r') as json_file:
        json_list = list(json_file)

    print(f'Reading dialogues for {split} split...')
    for json_str in json_list:
        datum = json.loads(json_str)
        datum['context'] = datum['dialogue'][0:-1]
        datum['response'] = {
            'valid': datum['dialogue'][-1],
            'invalid': datum['negations'][-1]
        }
        datum['explanations'] = []
        del datum['dialogue']
        del datum['negations']
        data[datum['id']] = datum
    print(f'Reading dialogues for {split} split...Done')

    # get explanations
    paths = Path(f'output/mturk/response/{split}').glob('dev[0-9]*.csv')
    for path in paths:
        with path.open() as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                turns = len([int(key.strip('Input.turn')) for key in row.keys() if 'Input.turn' in key])
                exp = {
                    'text': json.loads(row['Answer.taskAnswers'])[0]['explanation'],
                    'source': 'mturk'
                }
                id = int(row['Input.id'])
                data[id]['explanations'].append(exp)


    # write new data
    with open(outfile, 'w') as json_file:
        for datum in data.values():
            json.dump(datum, json_file)
            json_file.write('\n')

if __name__ == "__main__":
    # splits = ['train','dev','test']
    splits = ['dev']
    for sp in splits:
        # preprocess_binary_classification(sp)
        add_explanations(sp)