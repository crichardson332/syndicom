import json
from pathlib import Path
import pdb
from pprint import pprint

def toggle_speaker(speaker):
    if speaker == 'A':
        return 'B'
    else:
        return 'A'

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



if __name__ == '__main__':
    # display_annotations('whatsapp-rewrites2', dialogue_turns=4)
    display_annotations('whatsapp-rewrites3')