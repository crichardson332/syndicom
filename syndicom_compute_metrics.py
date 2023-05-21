import json

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
import torch
import json
from torch.optim import Adam
import numpy as np
from pprint import pprint

import numpy as np
import string 
import evaluation 
from rouge_score import rouge_scorer
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--split', choices=['dev', 'test'], help='Evaluate dev/test set responses', default='dev')
parser.add_argument('--rouge', action='store_true', help='Compute ROUGE1, ROUGE2, ROUGEL')
parser.add_argument('--sacrebleu', action='store_true', help='Compute Sacrebleu')
parser.add_argument('--meteor', action='store_true', help='compute Meteor')
parser.add_argument('--bertscore', action='store_true', help='compute BERTScore')
parser.add_argument('--all', action='store_true', help='compute everything')

args = parser.parse_args()

if args.all:
    args.rouge = True
    args.sacrebleu = True 
    args.meteor = True 
    args.bertscore = True 

# split = args.split
split = 'test'

model = 'gpt-ft-direct'

# if split == 'test':
#     lines = []
#     for line in open('syndicom_responses_test.json', 'r'):
#         lines.append(json.loads(line))    
# elif split == 'dev':
#     lines = []
#     for line in open('syndicom_responses_dev.json', 'r'):
#         lines.append(json.loads(line))    



# Load the gold response
datapath = '.'
infile = f'output/dialogue_modeling/fb_correction/{split}.jsonl'
with open(infile, 'r') as json_file:
    json_list = list(json_file)

json_strs_list = list(map(lambda x: json.loads(x), json_list))
# reference_responses = list(map(lambda x: x['response']['valid'], json_strs_list))
reference_responses = list(map(lambda x: [resp['text'] for resp in x['response'] if resp['source'] == 'valid'][0], json_strs_list))
predicted_responses = list(map(lambda x: [resp['text'] for resp in x['response'] if resp['source'] == model][0], json_strs_list))


if args.sacrebleu:
    from evaluate import load 
    sacrebleu = evaluate.load("sacrebleu")
    results = sacrebleu.compute(predictions=predicted_responses, references=reference_responses)
    print("SacreBLEU score is", results['score'])

if args.rouge:
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = list(map(lambda x: rouge.score(predicted_responses[x], reference_responses[x]), np.arange(len(predicted_responses))))
    precisions = list(map(lambda x: x['rouge1'].precision, rouge_scores))
    precisions_rouge2 = list(map(lambda x: x['rouge2'].precision, rouge_scores))
    precisions_rougeL = list(map(lambda x: x['rougeL'].precision, rouge_scores))
    print("rouge 1 precisions", np.mean(precisions))
    print("rouge 2 precisions", np.mean(precisions_rouge2))
    print("rouge L precisions", np.mean(precisions_rougeL))

    recalls_rouge1 = list(map(lambda x: x['rouge1'].recall, rouge_scores))
    recalls_rouge2 = list(map(lambda x: x['rouge2'].recall, rouge_scores))
    recalls_rougeL = list(map(lambda x: x['rougeL'].recall, rouge_scores))
    print("rouge 1 recalls", np.mean(recalls_rouge1))
    print("rouge 2 recalls", np.mean(recalls_rouge2))
    print("rouge L recalls", np.mean(recalls_rougeL))



    rouge = evaluation.load("rouge")
    results = rouge.compute(predictions=predicted_responses, references=reference_responses)
    print(results)

if args.bertscore:
    bertscore = load("bertscore")
    bertscore_results = bertscore.compute(predictions=predicted_responses, references=reference_responses, lang="en")
    print("BERTScore is", np.mean(bertscore_results['f1']))

if args.meteor:
    meteor = evaluation.load('meteor')
    results = meteor.compute(predictions=predicted_responses, references=reference_responses)
    print("Meteor is", results)