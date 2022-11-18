import transformers
import torch


from transformers import AutoModel, AutoTokenizer
from torch.nn import CosineSimilarity

# Load the model and tokenizer
model = AutoModel.from_pretrained("aws-ai/dse-bert-base")
tokenizer = AutoTokenizer.from_pretrained("aws-ai/dse-bert-base")

# Define the sentences of interests
texts = ["When will I get my card?",
         "Is there a way to know when my card will arrive?"]

# Define a function that calculate text embedding for a list of texts
def get_average_embedding(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Calculate the sentence embeddings by averaging the embeddings of non-padding words
    with torch.no_grad():
        embeddings = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        embeddings = torch.sum(embeddings[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return embeddings
  
cosine_sim = torch.nn.CosineSimilarity(dim=1)
embeddings = get_average_embedding(texts)

print("Similarity of the two sentences is: ", cosine_sim(embeddings[0], embeddings[1]).item())