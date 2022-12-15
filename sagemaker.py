import json
import pdb
import random
from pathlib import Path
from pprint import pprint

def write_sagemaker_manifest(split, num_samples=1000):
    infile = f'output/dataset/{split}.jsonl'
    with open(infile, 'r') as json_file:
        json_list = list(json_file)

    outfile = f'output/sagemaker/sample.jsonl'
    with open(outfile, 'w') as f:
        pass

    for json_str in json_list[0:num_samples]:
        datum = json.loads(json_str)
        dialogue = datum['dialogue'][0:-1] + [datum['negations'][-1]] 

        output = {
            'source': '[SEP]'.join(dialogue)
        }
        with open(outfile, 'a') as f:
            json.dump(output, f)
            f.write('\n')

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


if __name__ == "__main__":
    # splits = ['train', 'dev', 'test']
    splits = ['train']
    for sp in splits:
        # write_sagemaker_manifest(sp)
        # display_explanations('text-message-explanations')
        display_explanations('teach-ai-dialogues')