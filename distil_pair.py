# Andreas Goulas <goulasand@gmail.com>

import logging
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from sklearn.metrics import f1_score, classification_report
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import click

from models import BiLstmPair
from utils import normalize_text, load_vocab, tokenize, soft_ce

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Example:
    def __init__(self, sent0, sent1, label):
        self.sent0 = sent0
        self.sent1 = sent1
        self.label = label

class Features:
    def __init__(self, sent0, seq_len0, sent1, seq_len1, label):
        self.sent0 = sent0
        self.seq_len0 = seq_len0
        self.sent1 = sent1
        self.seq_len1 = seq_len1
        self.label = label

def convert_example(example, w2i, max_seq_len):
    sent0 = normalize_text(example.sent0)
    sent1 = normalize_text(example.sent1)
    ids0 = tokenize(sent0, w2i)[:max_seq_len]
    ids1 = tokenize(sent1, w2i)[:max_seq_len]
    return Features(
        ids0 + [0] * (max_seq_len - len(ids0)), len(ids0),
        ids1 + [0] * (max_seq_len - len(ids1)), len(ids1),
        example.label)

def get_tensor_dataset(features, is_augmented=False):
    label_type = torch.float32 if is_augmented else torch.long
    sents0 = torch.tensor([x.sent0 for x in features], dtype=torch.long)
    seq_lens0 = torch.tensor([x.seq_len0 for x in features], dtype=torch.long)
    sents1 = torch.tensor([x.sent1 for x in features], dtype=torch.long)
    seq_lens1 = torch.tensor([x.seq_len1 for x in features], dtype=torch.long)
    labels = torch.tensor([x.label for x in features], dtype=label_type)
    return TensorDataset(sents0, seq_lens0, sents1, seq_lens1, labels)

class XnliLoader:
    LABELS = ['neutral', 'contradiction', 'entailment']
    L2I = {'neutral': 0, 'contradiction': 1, 'contradictory': 1, 'entailment': 2}

    def load_features(self, filename, tokenizer, max_seq_len, is_augmented=False):
        features = []
        df = pd.read_csv(filename, sep='\t')
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='loading data'):
            if is_augmented:
                label = []
                for i in range(len(self.LABELS)):
                    label.append(float(row['score{}'.format(i)]))
            else:
                label = self.L2I[row['label']]
            example = Example(row['premise'], row['hypo'], label)
            features.append(convert_example(example, tokenizer, max_seq_len))
        return features

class SnliLoader:
    LABELS = ['neutral', 'contradiction', 'entailment']
    L2I = {label: i for i, label in enumerate(LABELS)}

    def load_features(self, filename, tokenizer, max_seq_len, is_augmented=False):
        features = []
        df = pd.read_csv(filename, sep='\t')
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='loading data'):
            label = self.L2I.get(row['gold_label'], 0)
            try:
                example = Example(row['sentence1'], row['sentence2'], label)
                features.append(convert_example(example, tokenizer, max_seq_len))
            except:
                pass
        return features

def evaluate(dataset, model, device, batch_size, target_names=None):
    loader = DataLoader(dataset, sampler=SequentialSampler(dataset),
        batch_size=batch_size)

    y_true = None
    y_pred = None

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = tuple(x.to(device) for x in batch)
            sents0, seq_lens0, sents1, seq_lens1, label_ids = batch
            logits = model(sents0, seq_lens0, sents1, seq_lens1)

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

@click.group()
def cli():
    pass

@cli.command('train')
@click.option('--task', required=True, type=click.Choice(['xnli', 'snli']), help='task name')
@click.option('--save_path', required=True, help='save path for  model checkpoint')
@click.option('--vectors', 'vectors_path', required=True, help='word embeddings file path')
@click.option('--train_dataset', 'train_path', required=True, help='train dataset file path')
@click.option('--val_dataset', 'val_path', required=True, help='validation dataset file path')
@click.option('--test_dataset', 'test_path', required=True, help='test dataset file path')
@click.option('--embed_dim', type=int, default=300, help='word embeddings dimension')
@click.option('--hidden_dim', type=int, default=512, help='lstm hidden layer size')
@click.option('--dropout', type=float, default=0.1, help='dropout rate')
@click.option('--fc_dim', type=int, default=256, help='fc layer size')
@click.option('--lstm_layers', type=int, default=1, help='number of lstm layers')
@click.option('--batch_size', type=int, default=256, help='batch size')
@click.option('--eval_batch_size', type=int, default=256, help='batch size for evaluation')
@click.option('--max_seq_len', type=int, default=64, help='max sequence length')
@click.option('--lr', type=float, default=1e-3, help='learning rate')
@click.option('--patience', type=int, default=3, help='number of epochs with no improvement')
@click.option('--seed', type=int, default=0, help='random seed')
@click.option('--is_augmented', is_flag=True, help='whether the training dataset is augmented')
@click.option('--use_mse', is_flag=True, help='use mean square error loss')
def cmd_train(task, save_path, vectors_path, train_path, val_path, test_path, embed_dim, hidden_dim, dropout,
        fc_dim, lstm_layers, batch_size, eval_batch_size, max_seq_len, lr, patience, seed, is_augmented, use_mse):
    device = torch.device('cuda:0')
    torch.manual_seed(seed)
    np.random.seed(seed)

    if task == 'xnli':
        data = XnliLoader()
    elif task == 'snli':
        data = SnliLoader()

    w2i, emb_matrix = load_vocab(vectors_path, embed_dim)
    emb_matrix = torch.from_numpy(emb_matrix)

    num_labels = len(data.LABELS)
    model = BiLstmPair(embed_dim, hidden_dim, fc_dim, dropout,
        lstm_layers, num_labels, emb_matrix).to(device)

    train_features = data.load_features(train_path, w2i, max_seq_len, is_augmented)
    train_dataset = get_tensor_dataset(train_features, is_augmented)

    val_features = data.load_features(val_path, w2i, max_seq_len)
    val_dataset = get_tensor_dataset(val_features)

    test_features = data.load_features(test_path, w2i, max_seq_len)
    test_dataset = get_tensor_dataset(test_features)

    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
        batch_size=batch_size)

    opt = Adam(model.parameters(), lr=lr)
    if is_augmented:
        if use_mse:
            crit = nn.MSELoss()
        else:
            crit = soft_ce
    else:
        crit = nn.CrossEntropyLoss()

    epoch = 0
    best_score = 0
    epochs_since = 0
    while epochs_since < patience:
        epoch_loss = 0
        model.train()
        for batch in train_loader:
            batch = tuple(x.to(device) for x in batch)
            sents0, seq_lens0, sents1, seq_lens1, label_ids = batch

            opt.zero_grad()
            logits = model(sents0, seq_lens0, sents1, seq_lens1)
            loss = crit(logits, label_ids)
            loss.backward()

            epoch_loss += loss.item()
            opt.step()

        val_score = evaluate(val_dataset, model, device, eval_batch_size)
        if val_score > best_score or epoch == 0:
            torch.save(model.state_dict(), save_path)
            best_score = val_score
            epochs_since = 0
        else:
            epochs_since += 1

        epoch_loss /= len(train_loader)
        logger.info('[epoch %d] loss = %f, val macro-f1 = %.2f%%',
            epoch + 1, epoch_loss, 100 * val_score)
        epoch += 1

    model.load_state_dict(torch.load(save_path))

    test_score = evaluate(test_dataset, model, device, eval_batch_size)
    logger.info('dev macro-f1 = %.2f%%\ntest macro-f1 = %.2f%%', 100 * best_score, 100 * test_score)

@cli.command('eval')
@click.option('--task', required=True, type=click.Choice(['xnli', 'snli']), help='task name')
@click.option('--checkpoint', required=True, help='path to model checkpoint')
@click.option('--vectors', 'vectors_path', required=True, help='word embeddings file path')
@click.option('--dataset', 'dataset_path', required=True, help='validation dataset file path')
@click.option('--embed_dim', type=int, default=300, help='word embeddings dimension')
@click.option('--hidden_dim', type=int, default=512, help='lstm hidden layer size')
@click.option('--dropout', type=float, default=0.1, help='dropout rate')
@click.option('--fc_dim', type=int, default=256, help='fc layer size')
@click.option('--lstm_layers', type=int, default=1, help='number of lstm layers')
@click.option('--batch_size', type=int, default=256, help='batch size')
@click.option('--max_seq_len', type=int, default=64, help='max sequence length')
def cmd_eval(task, checkpoint, vectors_path, dataset_path, embed_dim, hidden_dim, dropout,
        fc_dim, lstm_layers, batch_size, max_seq_len):
    device = torch.device('cuda:0')

    if task == 'xnli':
        data = XnliLoader()
    elif task == 'snli':
        data = SnliLoader()

    w2i, emb_matrix = load_vocab(vectors_path, embed_dim)
    emb_matrix = torch.from_numpy(emb_matrix)

    num_labels = len(data.LABELS)
    model = BiLstmPair(embed_dim, hidden_dim, fc_dim, dropout, lstm_layers,
        num_labels, emb_matrix).to(device)
    model.load_state_dict(torch.load(checkpoint), strict=False)

    features = data.load_features(dataset_path, w2i, max_seq_len)
    dataset = get_tensor_dataset(features)
    evaluate(dataset, model, device, batch_size, target_names=data.LABELS)

if __name__ == '__main__':
    cli()
