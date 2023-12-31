## SYNDICOM - Synthetic Dialogues with Commonsense
This repo contains the code used for the SigDial 2023 paper **SYNDICOM: Improving Conversational Commonsense with Error-Injection and Natural Language Feedback**
    
Paper: https://arxiv.org/pdf/2309.10015.pdf

Google scholar: https://scholar.google.com/citations?hl=en&user=6Lk0excAAAAJ 

SigDial 2023: https://2023.sigdial.org/

### Install

    pip install -r requirements.txt
    python -m spacy download en_core_web_sm

### Get ATOMIC-2020

    wget https://ai2-atomic.s3-us-west-2.amazonaws.com/data/atomic2020_data-feb2021.zip .
    unzip https://ai2-atomic.s3-us-west-2.amazonaws.com/data/atomic2020_data-feb2021.zip

### Dataset
Data files are available [here](dataset/syndicom). Each line of the `jsonl` data files contains:
- The template used to generate the dialogue
- Dialogue context (first N-1 turns)
- Responses, both valid and invalid (N'th turn)
- Crowd worker written feedback for the invalid turn (two per datum)
