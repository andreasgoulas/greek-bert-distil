# Andreas Goulas <goulasand@gmail.com>

import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from sklearn.metrics import f1_score, classification_report
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from bilstm import BiLstm
from utils import normalize_text, load_vocab, tokenize, soft_ce

logger = logging.getLogger(__name__)

class Example:
    def __init__(self, sent, label):
        self.sent = sent
        self.label = label

class Features:
    def __init__(self, sent, seq_len, label):
        self.sent = sent
        self.seq_len = seq_len
        self.label = label

def convert_example(example, w2i, max_seq_len):
    sent = normalize_text(example.sent)
    ids = tokenize(sent, w2i)[:max_seq_len]
    return Features(ids + [0] * (max_seq_len - len(ids)), len(ids), example.label)

def get_tensor_dataset(features, is_augmented=False):
    label_type = torch.float32 if is_augmented else torch.long
    sents = torch.tensor([x.sent for x in features], dtype=torch.long)
    seq_lens = torch.tensor([x.seq_len for x in features], dtype=torch.long)
    labels = torch.tensor([x.label for x in features], dtype=label_type)
    return TensorDataset(sents, seq_lens, labels)

class MakLoader:
    LABELS = ['Αθλητικά', 'Ρεπορτάζ', 'Οικονομία', 'Πολιτική', 'Διεθνή',
        'Τηλεόραση', 'Τέχνες-Πολιτισμός']
    L2I = {label: i for i, label in enumerate(LABELS)}

    def load_features(self, filename, w2i, max_seq_len, is_augmented=False):
        features = []
        df = pd.read_csv(filename)
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='loading data'):
            if is_augmented:
                label = []
                for i in range(len(self.LABELS)):
                    label.append(float(row['score{}'.format(i)]))
            else:
                label = self.L2I[row['Label']]
            example = Example(row['Text'], label)
            features.append(convert_example(example, w2i, max_seq_len))
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
            sents, seq_lens, labels = batch
            logits = model(sents, seq_lens)

            if y_true is None:
                y_true = labels.detach().cpu().numpy()
                y_pred = logits.detach().cpu().numpy()
            else:
                y_true = np.append(y_true, labels.detach().cpu().numpy(), axis=0)
                y_pred = np.append(y_pred, logits.detach().cpu().numpy(), axis=0)

    y_pred = y_pred.argmax(axis=-1)
    if target_names is not None:
        print(classification_report(y_true, y_pred, digits=3, target_names=target_names))

    return f1_score(y_true, y_pred, average='macro')

def run(args, model, train_dataset, val_dataset, device, silent=True):
    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size)

    opt = Adam(model.parameters(), lr=args.lr)
    if args.is_augmented:
        if args.use_mse:
            crit = nn.MSELoss()
        else:
            crit = soft_ce
    else:
        crit = nn.CrossEntropyLoss()

    epoch = 0
    best_score = 0
    epochs_since = 0
    while epochs_since < args.patience:
        epoch_loss = 0
        model.train()
        for batch in train_loader:
            batch = tuple(x.to(device) for x in batch)
            sents, seq_lens, label_ids = batch

            opt.zero_grad()
            logits = model(sents, seq_lens)
            loss = crit(logits, label_ids)
            loss.backward()

            epoch_loss += loss.item()
            opt.step()

        val_score = evaluate(args, val_dataset, model, device)
        if val_score > best_score or epoch == 0:
            torch.save(model.state_dict(), args.save_path)
            best_score = val_score
            epochs_since = 0
        else:
            epochs_since += 1

        epoch_loss /= len(train_loader)
        if not silent:
            logger.info('[epoch %d] loss = %f, val macro-f1 = %.2f%%',
                epoch + 1, epoch_loss, 100 * val_score)
        epoch += 1

    model.load_state_dict(torch.load(args.save_path))
    return best_score

def main_train():
    parser = argparse.ArgumentParser(description='greek bert distillation')
    parser.add_argument('--task', required=True, choices=['mak'], help='task name')
    parser.add_argument('--pretrained_bilstm', help='pretrained bilstm path')
    parser.add_argument('--save_path', default='bilstm_weights.pt', help='weights save file path')
    parser.add_argument('--vectors_file', required=True, help='word embeddings file path')
    parser.add_argument('--embed_dim', type=int, default=300, help='word embeddings dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='lstm hidden layer size')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--fc_dim', type=int, default=256, help='fc layer size')
    parser.add_argument('--lstm_layers', type=int, default=1, help='number of lstm layers')
    parser.add_argument('--train_dataset', required=True, help='train dataset file path')
    parser.add_argument('--val_dataset', required=True, help='validation dataset file path')
    parser.add_argument('--test_dataset', required=True, help='test dataset file path')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--patience', type=int, default=3, help='number of epochs with no improvement')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--is_augmented', action='store_true', help='whether the training dataset is augmented')
    parser.add_argument('--use_mse', action='store_true', help='use mean square error loss')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    device = torch.device('cuda:0')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.task == 'mak':
        data = MakLoader()

    w2i, emb_matrix = load_vocab(args.vectors_file, args.embed_dim)
    emb_matrix = torch.from_numpy(emb_matrix)

    num_labels = len(data.LABELS)
    model = BiLstm(args.embed_dim, args.hidden_dim, args.fc_dim,
        args.dropout, args.lstm_layers, num_labels, emb_matrix).to(device)
    if args.pretrained_bilstm:
        checkpoint = torch.load(args.pretrained_bilstm)
        model.rnn.load_state_dict(checkpoint['rnn_state_dict'])
        logger.info('loaded pretrained weights')

    train_features = data.load_features(
        args.train_dataset, w2i, args.max_seq_len, args.is_augmented)
    train_dataset = get_tensor_dataset(train_features, args.is_augmented)

    val_features = data.load_features(args.val_dataset, w2i, args.max_seq_len)
    val_dataset = get_tensor_dataset(val_features)

    test_features = data.load_features(args.test_dataset, w2i, args.max_seq_len)
    test_dataset = get_tensor_dataset(test_features)

    dev_score = run(args, model, train_dataset, val_dataset, device, silent=False)

    test_score = evaluate(args, test_dataset, model, device, target_names=data.LABELS)
    logger.info('dev macro-f1 = %.2f%%\ntest macro-f1 = %.2f%%', 100 * dev_score, 100 * test_score)

def main_eval():
    parser = argparse.ArgumentParser(description='greek bert distillation')
    parser.add_argument('--task', required=True, choices=['mak'], help='task name')
    parser.add_argument('--model', required=True, help='weights save file path')
    parser.add_argument('--vectors_file', required=True, help='word embeddings file path')
    parser.add_argument('--embed_dim', type=int, default=300, help='word embeddings dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='lstm hidden layer size')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--fc_dim', type=int, default=256, help='fc layer size')
    parser.add_argument('--lstm_layers', type=int, default=1, help='number of lstm layers')
    parser.add_argument('--dataset', required=True, help='validation dataset file path')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    device = torch.device('cuda:0')

    if args.task == 'mak':
        data = MakLoader()

    w2i, emb_matrix = load_vocab(args.vectors_file, args.embed_dim)
    emb_matrix = torch.from_numpy(emb_matrix)

    num_labels = len(data.LABELS)
    model = BiLstm(args.embed_dim, args.hidden_dim, args.fc_dim,
        args.dropout, args.lstm_layers, num_labels, emb_matrix).to(device)
    model.load_state_dict(torch.load(args.model), strict=False)

    features = data.load_features(args.dataset, w2i, args.max_seq_len)
    dataset = get_tensor_dataset(features)
    evaluate(args, dataset, model, device, target_names=data.LABELS)

def main_tune():
    parser = argparse.ArgumentParser(description='greek bert distillation')
    parser.add_argument('--pretrained_bilstm', help='pretrained bilstm path')
    parser.add_argument('--task', required=True, choices=['mak'], help='task name')
    parser.add_argument('--save_path', default='bilstm_weights.pt', help='weights save file path')
    parser.add_argument('--vectors_file', required=True, help='word embeddings file path')
    parser.add_argument('--embed_dim', type=int, default=300, help='word embeddings dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='lstm hidden layer size')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--fc_dim', type=int, default=256, help='fc layer size')
    parser.add_argument('--lstm_layers', type=int, default=1, help='number of lstm layers')
    parser.add_argument('--train_dataset', required=True, help='train dataset file path')
    parser.add_argument('--val_dataset', required=True, help='validation dataset file path')
    parser.add_argument('--test_dataset', required=True, help='test dataset file path')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--patience', type=int, default=3, help='number of epochs with no improvement')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--is_augmented', action='store_true', help='whether the training dataset is augmented')
    parser.add_argument('--use_mse', action='store_true', help='use mean square error loss')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')
    args = parser.parse_args()

    save_path = args.save_path
    logging.basicConfig(level=logging.INFO)

    device = torch.device('cuda:0')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.task == 'mak':
        data = MakLoader()

    w2i, emb_matrix = load_vocab(args.vectors_file, args.embed_dim)
    emb_matrix = torch.from_numpy(emb_matrix)

    train_features = data.load_features(
        args.train_dataset, w2i, args.max_seq_len, args.is_augmented)
    train_dataset = get_tensor_dataset(train_features, args.is_augmented)

    val_features = data.load_features(args.val_dataset, w2i, args.max_seq_len)
    val_dataset = get_tensor_dataset(val_features)

    test_features = data.load_features(args.test_dataset, w2i, args.max_seq_len)
    test_dataset = get_tensor_dataset(test_features)

    best_score = 0
    for lr in [5e-03, 1e-03, 5e-04, 3e-04, 2e-03]:
        for _ in range(10):
            num_labels = len(data.LABELS)
            model = BiLstm(args.embed_dim, args.hidden_dim, args.fc_dim,
                args.dropout, args.lstm_layers, num_labels, emb_matrix).to(device)
            if args.pretrained_bilstm:
                checkpoint = torch.load(args.pretrained_bilstm)
                model.rnn.load_state_dict(checkpoint['rnn_state_dict'])
                logger.info('loaded pretrained weights')

            args.lr = lr
            args.save_path = 'tmp_weights_%.2e.pt' % lr
            dev_score = run(args, model, train_dataset, val_dataset, device)
            logger.info('(lr=%f) dev macro-f1 = %.2f%%', lr, 100 * dev_score)
            if dev_score > best_score:
                torch.save(model.state_dict(), save_path)
                best_score = dev_score

    model.load_state_dict(torch.load(save_path))
    test_score = evaluate(args, test_dataset, model, device, target_names=data.LABELS)
    logger.info('dev macro-f1 = %.2f%%\ntest macro-f1 = %.2f%%', 100 * best_score, 100 * test_score)

if __name__ == '__main__':
    #main_train()
    #main_eval()
    main_tune()
