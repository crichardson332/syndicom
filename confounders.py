import json
import pdb
import random
from pprint import pprint
from tqdm import tqdm
import openai
import wandb


def gpt_negations(split):


    # create/overwrite output file
    outfile = f'output/dataset/tail_negations_{split}.jsonl'
    with open(outfile, 'w') as f:
        pass

    GPT_prompt = "Write the semantic opposite or antonym of each statement."

    ############################################
    ### TODO read in examples
    # read negation examples
    examples_file = f'examples/negation_examples.txt'
    with open(examples_file) as f:
        lines = f.readlines()
    ############################################

    # parse through all tail statements
    with open(f'output/dataset/tails_{split}.jsonl', 'r') as json_file:
        json_list = list(json_file)

    print(f'Running samples through GPT...')
    for i,json_str in enumerate(tqdm(json_list)):
        datum = json.loads(json_str)

        # TODO use GPT to negate datum['text']
        # negation = <gpt output>
        negation = "???"

        negations = {
            datum['text']: negation
        }

        # write example to file
        with open(outfile, 'a') as f:
            json.dump(negations, f)

    print(f'Running samples through GPT...Done')


if __name__ == "__main__":
    splits = ['train']
    for sp in splits:
        gpt_negations(sp)