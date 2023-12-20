import json
import csv
import util
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


def add_feedback(split):
    # output file create/overwrite file
    outfile = f'output/feedback/{split}.jsonl'

    # initialize final data dict
    data = {}

    # get dialogues
    infile_dlg = f'output/dataset/{split}.jsonl'
    with open(infile_dlg, 'r') as json_file:
        json_list = list(json_file)

    print(f'Reading dialogues for {split} split...')
    for json_str in json_list:
        datum = json.loads(json_str)
        datum['feedback'] = []
        data[datum['id']] = datum
    print(f'Reading dialogues for {split} split...Done')

    # get feedback
    paths = Path(f'output/mturk/response/{split}').glob(f'{split}[0-9]*.csv')
    for path in paths:
        with path.open() as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                turns = len([int(key.strip('Input.turn')) for key in row.keys() if 'Input.turn' in key])
                fb = {
                    'text': json.loads(row['Answer.taskAnswers'])[0]['explanation'],
                    'source': 'mturk'
                }
                id = int(row['Input.id'])
                data[id]['feedback'].append(fb)


    # write new data
    with open(outfile, 'w') as json_file:
        for datum in data.values():
            if len(datum['feedback']) > 0:
                json.dump(datum, json_file)
                json_file.write('\n')

def reformat(split):
    # final resting place
    outfile = f'output/dataset/{split}.jsonl'
    with open(outfile, 'w') as json_file:
        pass

    # get dialogues in old format
    infile = f'output/dataset/old_format/{split}.jsonl'
    with open(infile, 'r') as json_file:
        json_list = list(json_file)

    print(f'Reading dialogues for {split} split...')
    for json_str in json_list:
        datum = json.loads(json_str)
        datum['context'] = datum['dialogue'][0:-1]
        datum['response'] = {
            'valid': datum['dialogue'][-1],
            'invalid': datum['negations'][-1]
        }
        del datum['dialogue']
        del datum['negations']

        # write new data
        with open(outfile, 'a') as json_file:
            json.dump(datum, json_file)
            json_file.write('\n')
    print(f'Reading dialogues for {split} split...Done')

def process_for_gpt_finetuning(sp):
    # final resting place
    outfile = f'output/finetune/feedback/{sp}.jsonl'
    with open(outfile, 'a') as json_file:
        pass

    # get dialogues with feedback
    infile = f'output/feedback/{sp}.jsonl'
    with open(infile, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        datum = json.loads(json_str)
        prompt = util.get_gpt_feedback_prompt(datum)
        for fb in datum['feedback']:
            output = {
                'prompt': prompt,
                'completion': ' ' + fb['text'] + '###',
            }

            # write new data
            with open(outfile, 'a') as json_file:
                json.dump(output, json_file)
                json_file.write('\n')

def process_for_correction(sp):
    # final resting place
    outfile = f'output/finetune/correction/{sp}.jsonl'
    with open(outfile, 'w') as json_file:
        pass

    # get dialogues with feedback
    infile = f'output/feedback/{sp}.jsonl'
    with open(infile, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        datum = json.loads(json_str)
        for pair in util.get_gpt_correction_pairs(datum):
            with open(outfile, 'a') as json_file:
                json.dump(pair, json_file)
                json_file.write('\n')

def process_for_finetune_negation(sp):
    # final resting place
    outfile = f'output/finetune/negation/{sp}.jsonl'
    with open(outfile, 'a') as json_file:
        pass

    # get dialogues 
    infile = f'output/dataset/{sp}.jsonl'
    with open(infile, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        datum = json.loads(json_str)
        prompt = [resp['text'] for resp in datum['response'] if resp['source'] == 'invalid'][0]
        ref = [resp['text'] for resp in datum['response'] if resp['source'] == 'valid'][0]
        output = {
            'prompt': prompt,
            'completion': ref + '###',
        }

        # write new data
        with open(outfile, 'a') as json_file:
            json.dump(output, json_file)
            json_file.write('\n')


if __name__ == "__main__":
    # splits = ['train','dev','test']
    splits = ['dev']
    for sp in splits:
        # preprocess_binary_classification(sp)
        # add_feedback(sp)
        # reformat(sp)
        # process_for_gpt_finetuning(sp)
        # process_for_correction(sp)
        # process_for_finetune_negation(sp)
        add_feedback(sp)