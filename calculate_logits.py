# Andreas Goulas <goulasand@gmail.com>

import argparse
import logging
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

class Example:
    def __init__(self, sent0, sent1):
        self.sent0 = sent0
        self.sent1 = sent1

class Features:
    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

def convert_example(example, tokenizer, max_seq_len):
    out = tokenizer(example.sent0, example.sent1, padding='max_length',
        max_length=max_seq_len, truncation=True)
    return Features(out['input_ids'], out['attention_mask'], out['token_type_ids'])

def get_tensor_dataset(features):
    input_ids = torch.tensor([x.input_ids for x in features], dtype=torch.long)
    input_masks = torch.tensor([x.input_mask for x in features], dtype=torch.bool)
    segment_ids = torch.tensor([x.segment_ids for x in features], dtype=torch.int)
    return TensorDataset(input_ids, input_masks, segment_ids)

class XnliLoader:
    LABELS = ['neutral', 'contradiction', 'entailment']
    L2I = {'neutral': 0, 'contradiction': 1, 'contradictory': 1, 'entailment': 2}

    def load_features(self, filename, tokenizer, max_seq_len):
        features = []
        df = pd.read_csv(filename, sep='\t')
        for i, row in tqdm(df.iterrows(), total=df.shape[0], desc='loading data'):
            example = Example(row['premise'], row['hypo'])
            features.append(convert_example(example, tokenizer, max_seq_len))
        return features

class MakLoader:
    LABELS = ['Αθλητικά', 'Ρεπορτάζ', 'Οικονομία', 'Πολιτική', 'Διεθνή',
        'Τηλεόραση', 'Τέχνες-Πολιτισμός']
    L2I = {label: i for i, label in enumerate(LABELS)}

    def load_features(self, filename, tokenizer, max_seq_len):
        features = []
        df = pd.read_csv(filename)
        for i, row in tqdm(df.iterrows(), total=df.shape[0], desc='loading data'):
            example = Example(row['Text'], None)
            features.append(convert_example(example, tokenizer, max_seq_len))
        return features

def main():
    parser = argparse.ArgumentParser(description='greek bert distillation')
    parser.add_argument('--pretrained_model', required=True, help='pretrained model directory')
    parser.add_argument('--task', required=True, choices=['xnli', 'mak'], help='task name')
    parser.add_argument('--dataset', required=True, help='dataset file path')
    parser.add_argument('--save_file', required=True, help='logits save file path')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--tsv', action='store_true', help='whether to output tsv data')
    args = parser.parse_args()

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
        args.pretrained_model, num_labels=num_labels).to(device)

    features = data.load_features(args.dataset, tokenizer, args.max_seq_len)
    dataset = get_tensor_dataset(features)
    loader = DataLoader(dataset, sampler=SequentialSampler(dataset),
        batch_size=args.batch_size)

    y_pred = None
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc='evaluating'):
            batch = tuple(x.to(device) for x in batch)
            input_ids, input_masks, segment_ids = batch
            logits = model(input_ids, attention_mask=input_masks,
                token_type_ids=segment_ids)[0]

            if y_pred is None:
                y_pred = logits.detach().cpu().numpy()
            else:
                y_pred = np.append(y_pred, logits.detach().cpu().numpy(), axis=0)

    out_data = {}
    for i in range(len(data.LABELS)):
        out_data['score{}'.format(i)] = y_pred[:, i]

    aug_df = pd.DataFrame.from_dict(out_data)
    if args.tsv:
        aug_df.to_csv(args.save_file, index=None, sep='\t')
    else:
        aug_df.to_csv(args.save_file, index=None)

if __name__ == '__main__':
    main()
