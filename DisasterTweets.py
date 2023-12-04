import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

import torch

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

df_ = pd.read_csv('train.csv')
df_2 = pd.read_csv('test.csv')
sub = pd.read_csv('sample_submission.csv')

df = df_[['text', 'target']].copy()
df2 = df_2[['text']].copy()


df['target'].hist()

df.columns = ['sentence', 'label']
df2.columns = ['sentence']

df.to_csv('data_train.csv', index = None)
df2.to_csv('data_test.csv', index = None)

from datasets import load_dataset
raw_dataset = load_dataset('csv', data_files='data_train.csv')
raw_test = load_dataset('csv', data_files='data_test.csv')

split = raw_dataset['train'].train_test_split(test_size = 0.3)

checkpoint = 'distilbert-base-cased'

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_fn(batch):
    return tokenizer(batch['sentence'], truncation = True)

tokenized_datasets = split.map(tokenize_fn, batched = True)

tokenized_datasets['test'][0]

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 2)

from torchinfo import summary

summary(model)

training_args = TrainingArguments(
    output_dir='training_dir',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    optim='adamw_torch'
    )


def compute_metrics(logits_and_labels):
    logits, labels = logits_and_labels
    predictions = np.argmax(logits, axis = -1)
    acc = np.mean(predictions == labels)
    f1 = f1_score(labels, predictions, average = 'macro')
    return{'accuracy': acc, 'f1': f1}

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
    )


trainer.train()


!ls training_dir

from transformers import pipeline

savedmodel = pipeline('text-classification',
                      model='C:/Users/omer_/Desktop/MyML Projects/Disaster Tweets/training_dir/checkpoint-334'
                      )


tokenized_test = raw_test.map(tokenize_fn, batched = True)
tokenized_test

test_pred = savedmodel(tokenized_test["train"]['sentence'])

def get_label(d):
    return int(d['label'].split('_')[1])

test_pred_int = [get_label(d) for d in test_pred]

sub['target'] = test_pred_int
sub.to_csv('sub.csv', index = None)
