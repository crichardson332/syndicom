import json
from pathlib import Path
import pdb
from pprint import pprint
from tabulate import tabulate
import argparse

def toggle_speaker(speaker):
    if speaker == 'A':
        return 'B'
    else:
        return 'A'

def parse_args():
    parser=argparse.ArgumentParser(description="Inspect generated data for syndicom.")
    parser.add_argument("--data", default='dialogues')
    parser.add_argument("--subdir", default='opposites')
    parser.add_argument("--dialogue_turns", default=4)
    args=parser.parse_args()
    return args

def display_annotations(subdir, dialogue_turns=4):
    directory = f'sagemaker/annotations/{subdir}'

    primfile = Path(f'{directory}/sample.jsonl')
    json_file = primfile.open('r')
    json_list = list(json_file)
    inputs = []
    for json_str in json_list:
        inputs.append(json.loads(json_str)['source'])

    # for json_str in json_list:
    #     result = json.loads(json_str)
    #     print(f"result: {result}")
    #     print(isinstance(result, dict))

    # iterate over files in
    # that directory
    files = Path(directory).glob('[0-9]*.json')
    for file in files:
        idx = int(file.name.split('.')[0])
        print(f'\n--- DIALOGUE {idx} ---')
        machine_dialogue = inputs[idx].split('[SEP]')

        print(f'Machine dialogue:')
        speaker = 'A'
        for turn in machine_dialogue:
            print(f'  {speaker}: {turn}')
            speaker = toggle_speaker(speaker)
        
        # show turker annotations 
        ann = json.load(file.open('r'))
        for i_ans,ans in enumerate(ann['answers']):
            print(f'\nTurker{i_ans} rewrite:')
            speaker = 'B'
            for i_turn in range(dialogue_turns):
                speaker = toggle_speaker(speaker)
                try:
                    print(f"  {speaker}: {ans['answerContent'][f'utterance{i_turn}']}")
                except KeyError:
                    print(f"  {speaker}: - turn missing -")
                    continue
    
def display_dialogues(split):
    infile = f'output/dataset/{split}.jsonl'
    with open(infile, 'r') as json_file:
        json_list = list(json_file)


    for json_str in json_list:
        datum = json.loads(json_str)
        template = '\n'.join(datum['template'])
        dialogue = '\n'.join(datum['dialogue'])
        idx = datum['id']
        valids = [u'\u2713'] * len(datum['template'])
        if 'idx_confounder' in datum.keys():
            idx_confounder = int(datum['idx_confounder'])
            # arrow = ''.join(['\r']*idx_confounder) + '<---'
            valids[idx_confounder] = 'x'
        valids = '\n'.join(valids)
        print(tabulate([[template, valids, dialogue, idx]], headers=['Template','Valid','Dialogue', 'ID'],tablefmt="simple_grid"))
        pdb.set_trace()

def display_templates(split):
    infile = f'output/templates/{split}.jsonl'
    with open(infile, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        datum = json.loads(json_str)
        template = '\n'.join(datum['template'])
        idx = datum['id']
        valids = [u'\u2713'] * len(datum['template'])
        if 'idx_confounder' in datum.keys():
            idx_confounder = int(datum['idx_confounder'])
            # arrow = ''.join(['\r']*idx_confounder) + '<---'
            valids[idx_confounder] = 'x'
        valids = '\n'.join(valids)
        print(tabulate([[template, valids, idx]], headers=['Template','Valid','ID'],tablefmt="simple_grid"))
        pdb.set_trace()


def main(args, split):
    if args.data == 'templates':
        display_templates(split)
    elif args.data == 'dialogues':
        display_dialogues(split)
    elif args.data == 'annotations':
        display_annotations(args.subdir, args.dialogue_turns)


if __name__ == '__main__':
    args = parse_args()
    main(args, 'train')