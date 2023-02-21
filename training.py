import csv
import pandas as pd
from transformers import RobertaTokenizerFast, Trainer, TrainingArguments, InputExample
from sklearn.model_selection import train_test_split

# Function to load the dataset
def load_data(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Create a list of InputExample objects
    examples = []
    for i, row in data.iterrows():
        text, label = row[0], row[1]
        examples.append(InputExample(guid=i, text_a=text, label=label))

    return examples

# Load the dataset
examples = load_data('./data.csv')
train_examples, eval_examples = train_test_split(examples, test_size=0.2, random_state=42)

# Load the RoBERTa-large tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')

# Define the model and the trainer
model = transformers.RobertaForSequenceClassification.from_pretrained('roberta-large')
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir='./results',
        evaluation_strategy='steps',
        eval_steps=1000,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    ),
    train_dataset=train_examples,
    eval_dataset=eval_examples,
    tokenizer=tokenizer
)

# Train the model
trainer.train()
