from transformers import AutoModelForSequenceClassification,AutoTokenizer
import torch
from datasets import load_dataset
import pandas as pd
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

data = pd.read_csv("./data/test.csv")
model = AutoModelForSequenceClassification.from_pretrained("/root/Project/ccf_train_book_label_predict/checkpoints/checkpoint-8650")
tokenizer = AutoTokenizer.from_pretrained(r"/root/Project/pretrained_models/rbt3")
device = "cuda" if torch.cuda.is_available() else "cpu"
datasets = load_dataset("csv", data_files="./data/test.csv", split="train")
model.to(device)

def process_function(examples):
    tokenized_examples = tokenizer(examples["text"], max_length=128, truncation=True)
    return tokenized_examples


tokenized_datasets = datasets.map(process_function, batched=True)

sen = data['text'].tolist()
ids = data['node_id'].tolist()
res = {"node_id": [], "label":[]}
model.eval()
with torch.inference_mode():
    for i in range(0, len(sen), 100):
        print(i)
        inputs = tokenizer(sen[i:i+100], max_length=128, truncation=True,padding=True,return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1)
        res['label'] += pred.tolist()
        res['node_id'] += ids[i:i+100]

pd.DataFrame(res).to_csv("./submission.csv")
