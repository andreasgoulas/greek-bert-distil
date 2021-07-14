# Andreas Goulas <goulasand@gmail.com>

import argparse
import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, BertTokenizer, BertForSequenceClassification
from sklearn.metrics import f1_score, classification_report
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

class Example:
    def __init__(self, sent0, sent1, label_id):
        self.sent0 = sent0
        self.sent1 = sent1
        self.label_id = label_id

class Features:
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def convert_example(example, tokenizer, max_seq_len):
    out = tokenizer(example.sent0, example.sent1, padding='max_length',
        max_length=max_seq_len, truncation=True)
    return Features(out['input_ids'], out['attention_mask'], out['token_type_ids'],
        example.label_id)

def get_tensor_dataset(features):
    input_ids = torch.tensor([x.input_ids for x in features], dtype=torch.long)
    input_masks = torch.tensor([x.input_mask for x in features], dtype=torch.bool)
    segment_ids = torch.tensor([x.segment_ids for x in features], dtype=torch.int)
    label_ids = torch.tensor([x.label_id for x in features], dtype=torch.long)
    return TensorDataset(input_ids, input_masks, segment_ids, label_ids)

class XnliLoader:
    LABELS = ['neutral', 'contradiction', 'entailment']
    L2I = {'neutral': 0, 'contradiction': 1, 'contradictory': 1, 'entailment': 2}

    def load_features(self, filename, tokenizer, max_seq_len):
        features = []
        df = pd.read_csv(filename, sep='\t')
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='loading data'):
            example = Example(row['premise'], row['hypo'], self.L2I[row['label']])
            features.append(convert_example(example, tokenizer, max_seq_len))
        return features

class MakLoader:
    LABELS = ['Αθλητικά', 'Ρεπορτάζ', 'Οικονομία', 'Πολιτική', 'Διεθνή',
        'Τηλεόραση', 'Τέχνες-Πολιτισμός']
    L2I = {label: i for i, label in enumerate(LABELS)}

    def load_features(self, filename, tokenizer, max_seq_len):
        features = []
        df = pd.read_csv(filename)
        for index, row in df.iterrows():
            example = Example(row['Text'], None, self.L2I[row['Label']])
            features.append(convert_example(example, tokenizer, max_seq_len))
        return features

def evaluate(args, dataset, model, device, target_names=None):
    loader = DataLoader(dataset, sampler=SequentialSampler(dataset),
        batch_size=args.batch_size)

    y_true = None
    y_pred = None

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = tuple(x.to(device) for x in batch)
            input_ids, input_masks, segment_ids, label_ids = batch
            logits = model(input_ids, attention_mask=input_masks,
                token_type_ids=segment_ids)[0]

            if y_true is None:
                y_true = label_ids.detach().cpu().numpy()
                y_pred = logits.detach().cpu().numpy()
            else:
                y_true = np.append(y_true, label_ids.detach().cpu().numpy(), axis=0)
                y_pred = np.append(y_pred, logits.detach().cpu().numpy(), axis=0)

    y_pred = y_pred.argmax(axis=-1)
    if target_names is not None:
        print(classification_report(y_true, y_pred, digits=3, target_names=target_names))

    return f1_score(y_true, y_pred, average='macro')

def main_train():
    parser = argparse.ArgumentParser(description='greek bert distillation')
    parser.add_argument('--save_dir', default='./greek-bert-nli', help='save directory for trained model')
    parser.add_argument('--task', required=True, choices=['xnli', 'mak'], help='task name')
    parser.add_argument('--train_dataset', required=True, help='train dataset file path')
    parser.add_argument('--val_dataset', required=True, help='validation dataset file path')
    parser.add_argument('--lr', type=float, default=5e-05, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs to train')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logging.basicConfig(level=logging.INFO)

    device = torch.device('cuda:0')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.task == 'xnli':
        data = XnliLoader()
    elif args.task == 'mak':
        data = MakLoader()

    num_labels = len(data.LABELS)
    tokenizer = BertTokenizer.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')
    model = BertForSequenceClassification.from_pretrained(
        'nlpaueb/bert-base-greek-uncased-v1', num_labels=num_labels).to(device)

    train_features = data.load_features(args.train_dataset, tokenizer, args.max_seq_len)
    train_dataset = get_tensor_dataset(train_features)

    val_features = data.load_features(args.val_dataset, tokenizer, args.max_seq_len)
    val_dataset = get_tensor_dataset(val_features)

    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size)

    opt = AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        epoch_loss = 0
        model.train()
        pbar = tqdm(train_loader, desc='finetuning')
        for step, batch in enumerate(pbar):
            batch = tuple(x.to(device) for x in batch)
            input_ids, input_masks, segment_ids, label_ids = batch

            opt.zero_grad()
            loss = model(input_ids, attention_mask=input_masks,
                token_type_ids=segment_ids, labels=label_ids)[0]
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        epoch_loss /= len(train_loader)
        logger.info('[epoch %d] loss = %f', epoch + 1, epoch_loss)
    
    model.save_pretrained(args.save_dir)
    val_score = evaluate(args, val_dataset, model, device, target_names=data.LABELS)
    logger.info('val macro-f1 = %.2f%%', 100 * val_score)

def main_eval():
    parser = argparse.ArgumentParser(description='greek bert distillation')
    parser.add_argument('--task', required=True, choices=['xnli', 'mak'], help='task name')
    parser.add_argument('--model', required=True, help='weights save file path')
    parser.add_argument('--dataset', required=True, help='dataset file path')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda:0')

    if args.task == 'xnli':
        data = XnliLoader()
    elif args.task == 'mak':
        data = MakLoader()

    num_labels = len(data.LABELS)
    tokenizer = BertTokenizer.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')
    model = BertForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels).to(device)

    features = data.load_features(args.dataset, tokenizer, args.max_seq_len)
    dataset = get_tensor_dataset(features)

    test_score = evaluate(args, dataset, model, device, target_names=data.LABELS)
    logger.info('test macro-f1 = %.2f%%', 100 * test_score)

if __name__ == '__main__':
    main_train()
    #main_eval()
