from datasets import load_dataset
import pdb

split = 'train'
input_data_path = f'output/dataset/{split}.jsonl'
dataset = load_dataset("json", data_files=input_data_path)

pdb.set_trace()