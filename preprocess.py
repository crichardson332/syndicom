import json

def preprocess_binary_classification(split):
    infile = f'output/dataset/{split}.jsonl'
    with open(infile, 'r') as json_file:
        json_list = list(json_file)

    outfile = f'output/huggingface/syndicom/{split}.jsonl'
    for json_str in json_list:
        datum = json.loads(json_str)
        context = '\n'.join(datum['dialogue'][0:-1])
        valid = context + '[SEP]' + datum['dialogue'][-1]
        invalid = context + '[SEP]' + datum['negations'][-1]
        dout1 = {
            'text': valid,
            'label': 1
        }
        dout2 = {
            'text': invalid,
            'label': 0
        }

        with open(outfile, 'a') as f:
            json.dump(f, dout1)
            f.write('\n')
            json.dump(f, dout2)
            f.write('\n')

if __name__ == "__main__":
    # splits = ['train','dev','test']
    splits = ['dev']
    for sp in splits:
        preprocess_binary_classification(sp)