from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorWithPadding
import torch
from sklearn.metrics import f1_score, accuracy_score

if torch.cuda.is_available():
    device=torch.device("cuda")
    print("使用gpu")
else:
    print("使用cpu")

dataset = load_dataset("csv", data_files="./data/train.csv", split="train")
dataset = dataset.filter(lambda x: x["label"] is not None)
datasets = dataset.train_test_split(test_size=0.1)

tokenizer = AutoTokenizer.from_pretrained(r"/root/Project/pretrained_models/rbt3")

def process_function(examples):
    tokenized_examples = tokenizer(examples["text"], max_length=128, truncation=True)
    tokenized_examples["labels"] = examples["label"]
    return tokenized_examples

tokenized_datasets = datasets.map(process_function, batched=True, remove_columns=datasets["train"].column_names)

model = AutoModelForSequenceClassification.from_pretrained(r"/root/Project/pretrained_models/rbt3",num_labels=24)


def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)
    res = {'acc':accuracy_score(labels, predictions),
           "f1":f1_score(labels, predictions,average='micro')}
    return res

train_args = TrainingArguments(output_dir="./checkpoints",      # 输出文件夹
                               per_device_train_batch_size=64,  # 训练时的batch_size
                               per_device_eval_batch_size=128,  # 验证时的batch_size
                               logging_steps=10,                # log 打印的频率
                               evaluation_strategy="epoch",     # 评估策略
                               save_strategy="epoch",           # 保存策略
                               save_total_limit=3,              # 最大保存数
                               learning_rate=2e-5,              # 学习率
                               weight_decay=0.01,               # weight_decay
                               metric_for_best_model="f1",      # 设定评估指标
                               load_best_model_at_end=True,
                               num_train_epochs=10)     # 训练完成后加载最优模型

trainer = Trainer(model=model,
                  args=train_args,
                  train_dataset=tokenized_datasets["train"],
                  eval_dataset=tokenized_datasets["test"],
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                  compute_metrics=eval_metric)
print("开始训练")
trainer.train()

print("开始验证")
print(trainer.evaluate(tokenized_datasets['test']))
