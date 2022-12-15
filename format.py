import json
import pdb
import random
from tqdm import tqdm
from pprint import pprint

def to_dialogue_explanations(split, num_samples=1000):
     
    infile = f'output/dataset/{split}.jsonl'
    outfile = f'sample.jsonl'

    # read templates and get GPT response
    with open(infile, 'r') as json_file:
        json_list = list(json_file)

    print(f'Generating sagemaker manifest for {split} split...')
    for json_str in tqdm(json_list[0:num_samples]):

if __name__ == "__main__":
    # splits = ['train', 'dev', 'test']
    splits = ['train']
    splits = ['dev', 'test']
    for sp in splits:
        def to_dialogue_explanations(split)
        # gen_templates(sp, write_tails=True, do_confounders=False)