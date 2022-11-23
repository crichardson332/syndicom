import json

def preprocess():
    # FIXME for now we have to create all splits from the train split
    # because we dont have all the data yet
    infile = f'output/dataset/train.jsonl'
    with open(infile, 'r') as json_file:
        json_list = list(json_file)

    outfile = f'output/huggingface/syndicom.jsonl'
    for json_str in json_list:
        datum = json.loads(json_str)