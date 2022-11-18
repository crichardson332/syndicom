import json
import pdb
import random
from pprint import pprint

def write_dialogue_turns(split):
    infile = f'output/dataset/{split}.jsonl'
    with open(infile, 'r') as json_file:
        json_list = list(json_file)

    outfile = f'output/sagemaker/sample.jsonl'
    with open(outfile, 'w') as f:
        pass

    for json_str in json_list[0:10]:
        datum = json.loads(json_str)
        dialogue = datum['dialogue']

        output = {
            'source': '[SEP]'.join(dialogue)
        }
        with open(outfile, 'a') as f:
            json.dump(output, f)
            f.write('\n')


if __name__ == "__main__":
    # splits = ['train', 'dev', 'test']
    splits = ['train']
    for sp in splits:
        write_dialogue_turns(sp)