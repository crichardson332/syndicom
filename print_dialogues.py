import json
from pprint import pprint

infile = 'train_negations.jsonl'
with open(infile, 'r') as json_file:
    json_list = list(json_file)

for json_str in json_list:
    datum = json.loads(json_str)
    print('---')
    pprint(datum)