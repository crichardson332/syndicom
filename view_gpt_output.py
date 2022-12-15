import json
import pdb

file_name = "/Users/crichardson8/gitrepos/syndicom/11102022_165514_gpt3_opposite_responses_progress.json"
data = json.load(open(file_name, "r"))
for i in range(len(data)):
    text = data[i][1] 
    negation = data[i][2].replace('\n','')
    print(f'text    : {text}') 
    print(f'negation: {negation}') 
    print('---')
