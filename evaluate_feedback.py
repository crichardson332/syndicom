import json

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
import torch
import json
from torch.optim import Adam
import numpy as np
from pprint import pprint
import pandas as pd

from evaluate import load 
import numpy as np
import string 
import evaluate 
from rouge_score import rouge_scorer
import argparse 


def scoring_function(predictions, references_1, references_2, metric):
    score_df = pd.DataFrame()

    if 'rouge' in metric:

        rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
        rouge_scores_with_ref_1 = list(map(lambda x: rouge.score(references_1[x], predictions[x]), np.arange(len(predictions))))
        score_df['score1'] = list(map(lambda x: x[metric].fmeasure, rouge_scores_with_ref_1))
    
        rouge_scores_with_ref_2 = list(map(lambda x: rouge.score(references_2[x], predictions[x]), np.arange(len(predictions))))
        score_df['score2'] = list(map(lambda x: x[metric].fmeasure, rouge_scores_with_ref_2))

    if metric == 'bertscore':
        bertscore = load("bertscore")
        bertscore_results_1 = bertscore.compute(predictions=predictions, references=references_1, lang="en")
        bertscore_results_2 = bertscore.compute(predictions=predictions, references=references_2, lang="en")

        score_df['score1'] = bertscore_results_1['f1']
        score_df['score2'] = bertscore_results_2['f1']

    if metric == 'sacrebleu':
        sacrebleu = evaluate.load('sacrebleu')

        score_1 = []
        score_2 = []
        score_df = pd.DataFrame()

        for i in tqdm(range(len(predictions))):
            pred_i = [predictions[i]]
            ref_1_i = [references_1[i]]
            ref_2_i = [references_2[i]]

            score_1.append(sacrebleu.compute(predictions=pred_i, references=ref_1_i)['score'])
            score_2.append(sacrebleu.compute(predictions=pred_i, references=ref_2_i)['score'])

        score_df['score1'] = score_1
        score_df['score2'] = score_2

    if metric == 'bleu':
        bleu = evaluate.load('bleu')

        score_1 = []
        score_2 = []
        score_df = pd.DataFrame()

        for i in tqdm(range(len(predictions))):
            pred_i = [predictions[i]]
            ref_1_i = [references_1[i]]
            ref_2_i = [references_2[i]]

            score_1.append(bleu.compute(predictions=pred_i, references=ref_1_i)['bleu'])
            score_2.append(bleu.compute(predictions=pred_i, references=ref_2_i)['bleu'])

        score_df['score1'] = score_1
        score_df['score2'] = score_2


    score_df['max_score'] = score_df[['score1', 'score2']].max(axis=1)
    score_df['min_score'] = score_df[['score1', 'score2']].min(axis=1)
    score_df['avg_score'] = score_df[['score1', 'score2']].mean(axis=1)

    return score_df['max_score'].mean(), score_df['min_score'].mean(), score_df['avg_score'].mean()

infile = 'dev_feedback.jsonl'

with open(infile, 'r') as json_file:
    json_list = list(json_file)

data = []

for i,json_str in enumerate(json_list):
    datum = json.loads(json_str)
    data.append(datum)

feedback = list(map(lambda x: x['feedback'], data))
feedback = list(filter(lambda x: len(x)==4, feedback))

feedback_df = pd.DataFrame()

gpt_turbo = list(map(lambda x: x[0]['text'], feedback))
gpt_ft = list(map(lambda x: x[1]['text'], feedback))
mturk_1 = list(map(lambda x: x[2]['text'], feedback))
mturk_2 = list(map(lambda x: x[3]['text'], feedback))

"""
scoring_function(gpt_turbo, mturk_1, mturk_2, metric='rouge1')
scoring_function(gpt_ft, mturk_1, mturk_2, metric='rouge1')

scoring_function(gpt_turbo, mturk_1, mturk_2, metric='rouge2')
scoring_function(gpt_ft, mturk_1, mturk_2, metric='rouge2')

scoring_function(gpt_turbo, mturk_1, mturk_2, metric='rougeL')
scoring_function(gpt_ft, mturk_1, mturk_2, metric='rougeL')

scoring_function(gpt_turbo, mturk_1, mturk_2, metric='bertscore')
scoring_function(gpt_ft, mturk_1, mturk_2, metric='bertscore')
"""

print("Sacrebleu")
max_score, min_score, avg_score = scoring_function(gpt_turbo, mturk_1, mturk_2, metric='sacrebleu')
print(f"GPT_Turbo max:{max_score}, min:{min_score}, avg:{avg_score}")
max_score,  min_score, avg_score = scoring_function(gpt_ft, mturk_1, mturk_2, metric='sacrebleu')
print(f"GPT FT max:{max_score}, min:{min_score}, avg:{avg_score}")


print("Bleu")
max_score,  min_score, avg_score = scoring_function(gpt_turbo, mturk_1, mturk_2, metric='bleu')
print(f"GPT_Turbo max:{max_score}, min:{min_score}, avg:{avg_score}")
max_score,  min_score, avg_score = scoring_function(gpt_ft, mturk_1, mturk_2, metric='bleu')
print(f"GPT FT max:{max_score}, min:{min_score}, avg:{avg_score}")