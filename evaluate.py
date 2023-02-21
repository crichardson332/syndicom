import transformers
import torch
import json
import pdb
from pprint import pprint
from tqdm import tqdm


from transformers import AutoModel, AutoTokenizer
from torch.nn import CosineSimilarity

def compute_score(context, response, model, tokenizer):
    # Load the model and tokenizer
    # model = AutoModel.from_pretrained("aws-ai/dse-bert-base")
    # tokenizer = AutoTokenizer.from_pretrained("aws-ai/dse-bert-base")

    # Define the sentences of interests
    texts = [context, response]

    # Define a function that calculate text embedding for a list of texts
    def get_average_embedding(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        # Calculate the sentence embeddings by averaging the embeddings of non-padding words
        with torch.no_grad():
            embeddings = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            embeddings = torch.sum(embeddings[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
            return embeddings
    
    cosine_sim = torch.nn.CosineSimilarity(dim=0)
    embeddings = get_average_embedding(texts)
    score = cosine_sim(embeddings[0], embeddings[1]).item()

    # print(f'{response}: score = {score}')
    return score

def eval_dse():
    infile = f'output/dataset/train.jsonl'
    with open(infile, 'r') as json_file:
        json_list = list(json_file)

    model = AutoModel.from_pretrained("aws-ai/dse-bert-base")
    tokenizer = AutoTokenizer.from_pretrained("aws-ai/dse-bert-base")

    n_total = 0
    n_correct = 0
    print(f'Evaluating DSE scores...')
    for json_str in tqdm(json_list[0:500]):
        datum = json.loads(json_str)

        for idx in range(2, len(datum['dialogue'])):
            scores = []
            context = '\n'.join(datum['dialogue'][0:idx-1])
            valid = datum['dialogue'][idx] 
            invalid = datum['negations'][idx] 

            score_valid = compute_score(context, valid, model, tokenizer)
            score_invalid = compute_score(context, invalid, model, tokenizer)

            n_total += 1
            if score_valid > score_invalid:
                n_correct += 1
            else:
                pprint(datum['dialogue'][0:idx-1])

    print(f'Evaluating DSE scores...Done')
    print(f'Accuracy: {n_correct/n_total*100}%')

    pdb.set_trace()



if __name__ == '__main__':
    eval_dse()
    # compute_score('I got an A on my test!', 'You got an A on your test!')
    # compute_score('I got an A on my test!', 'I live in a giant bucket.')

    # context = "I got accepted into college! That's great! I bet you can't wait to buy stuff for your dorm now. Yeah I'm excited about that. I wanted to go to college so bad."
    # # response = "It's a good thing you kept your grades up in high school."
    # response = "I guess it doesn't matter that you got low grades in high school!"
    # compute_score(context, response)