import json
import csv
import pdb
import random
import pdb
from math import inf
from pathlib import Path
from pprint import pprint

def gen_csv(split, num_samples=inf, num_turns=3):
    infile = f'output/dataset/{split}.jsonl'
    with open(infile, 'r') as json_file:
        json_list = list(json_file)

    outfile = f'output/mturk/query/{split}/{split}{num_turns}.csv'
    cw = csv.writer(open(outfile, 'w'))
    # write csv headers first
    headers = ['id']
    for i in range(num_turns):
        headers.append(f'turn{i}')
    cw.writerow(headers)

    count = 0
    for json_str in json_list:
        datum = json.loads(json_str)
        if (len(datum['context'])+1) != num_turns:
            continue

        output = [f'{datum["id"]}'] + datum['context'] 
        invr = [resp['text'] for resp in datum['response'] if resp['source'] == 'invalid'][0]
        output.append(invr)

        # write output string
        cw.writerow(output)

        count += 1
        if count >= num_samples:
            break

def gen_csv_full(split, num_samples=inf):
    infile = f'output/dataset/{split}.jsonl'
    with open(infile, 'r') as json_file:
        json_list = list(json_file)

    outfile = f'output/mturk/query/{split}.csv'
    cw = csv.writer(open(outfile, 'w'))
    # write csv headers first
    headers = ['id','num_turns']
    num_columns = 20
    for i in range(num_columns-2):
        headers.append(f'turn{i}')
    cw.writerow(headers)

    count = 0
    for json_str in json_list:
        datum = json.loads(json_str)
        num_turns = len(datum['dialogue'])

        # fill with padding first
        output = ['nodata'] * num_columns

        # 
        output[0] = f'{datum["id"]}'
        output[1] = f'{num_turns}'
        
        # write first N-1 turns
        for idx in range(num_turns-1):
            try:
                output[2+idx] = f'{datum["dialogue"][idx]}'
            except IndexError:
                pdb.set_trace()

        # last turn is the negation
        output[num_turns+1] = f'{datum["negations"][-1]}'

        # write output string
        cw.writerow(output)

        count += 1
        if count >= num_samples:
            break

def display_explanations(subdir):
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
    files = Path(directory).glob('[0-9]*/*.json')
    # pdb.set_trace()
    for file in files:
        # pdb.set_trace()
        idx = int(file.parent.name)
        print(f'\n--- DIALOGUE {idx} ---')
        machine_dialogue = inputs[idx].split('[SEP]')

        print(f'Dialogue:')
        for turn in machine_dialogue:
            print(f'  {turn}')
        
        # show turker annotations 
        ann = json.load(file.open('r'))
        for i_ans,ans in enumerate(ann['answers']):
            print(f'\nTurker explanation:')
            print(f"  {ans['answerContent'][f'explanation']}")
            # for i_turn in range(dialogue_turns):
            #     speaker = toggle_speaker(speaker)
            #     try:
            #         print(f"  {speaker}: {ans['answerContent'][f'explanation']}")
            #     except KeyError:
            #         print(f"  {speaker}: - turn missing -")
            #         continue

def gen_response_selection(split, num_samples=inf, num_turns=3):
    infile = f'output/dialogue_modeling/fb_correction/{split}.jsonl'
    with open(infile, 'r') as json_file:
        json_list = list(json_file)

    outfile = f'output/mturk/response_selection/query/{split}{num_turns}.csv'
    cw = csv.writer(open(outfile, 'w'))
    # write csv headers first
    headers = ['id']
    for i in range(num_turns-1):
        headers.append(f'turn{i}')
    headers += ['syndicom', 'chatgpt']
    cw.writerow(headers)

    count = 0
    for json_str in json_list:
        datum = json.loads(json_str)
        if (len(datum['context'])+1) != num_turns:
            continue

        output = [f'{datum["id"]}'] + datum['context'] 
        # add baseline and our responses
        syndicom = [resp['text'] for resp in datum['response'] if resp['source'] == 'gpt-ft-correction'][0].strip("\n")
        chatgpt = [resp['text'] for resp in datum['response'] if resp['source'] == 'gpt-3.5-turbo'][0].strip("\n")
        output += [syndicom, chatgpt]


        # write output string
        cw.writerow(output)

        count += 1
        if count >= num_samples:
            break


if __name__ == "__main__":
    # splits = ['train', 'dev', 'test']
    # splits = ['train']
    splits = ['test']
    # turns = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    turns = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    # turns = [3]
    for sp in splits:
        # write_sagemaker_manifest(sp)
        # display_explanations('text-message-explanations')
        # display_explanations('teach-ai-dialogues')
        for t in turns:
            # gen_csv(sp, num_turns=t)
            # gen_csv_full(sp, num_samples=100)
            # gen_csv(sp, num_turns=t)
            gen_response_selection(sp, num_turns=t)