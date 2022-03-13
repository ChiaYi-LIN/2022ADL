import os
import math
import collections
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

import numpy as np
import pandas as pd
from seqeval.metrics import accuracy_score

from dataset import SeqIOBDataset
from utils import Vocab
from model import SeqIOBTagger

# from torch.utils.tensorboard import SummaryWriter

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

#fix seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):
    # fix random seed
    same_seeds(args.seed)

    # vocab.pkl contains top 10000 fequently used words, and it is for token2idx
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
    # print(vocab.token2idx.items())

    # tag2idx.json is for tag2idx
    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())
    # print(tag2idx.items())
    
    # read in train.json and eval.json
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: pd.read_json(path) for split, path in data_paths.items()}
    
    weights = [0 for _ in range(len(tag2idx))]
    occur = data[TRAIN]['tags'].to_list() + data[DEV]['tags'].to_list()
    cnt_list = collections.Counter([x for sublist in occur for x in sublist])
    # print(cnt_list.items())
    for tag, cnt in cnt_list.items():
        weights[tag2idx[tag]] = cnt
    max_w = max(weights)
    for i in range(len(weights)):
        weights[i] = max_w / weights[i]
    weights = torch.FloatTensor(weights)
    # print(weights)
    
    # Load Dataset
    datasets: Dict[str, SeqIOBDataset] = {
        split: SeqIOBDataset(x=split_data[['tokens']], y=split_data[['tags']], vocab=vocab, label_mapping=tag2idx, max_len=args.max_len)
        for split, split_data in data.items()
    }
    # print(datasets['train'].x)
    # print(datasets['train'].y)

    # TODO: crecate DataLoader for train / dev datasets
    train_loader = DataLoader(datasets[TRAIN], batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=datasets[TRAIN].collate_fn)
    dev_loader = DataLoader(datasets[DEV], batch_size=args.batch_size, shuffle=False, pin_memory=True, collate_fn=datasets[DEV].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqIOBTagger(
        embeddings=embeddings,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        num_class=len(tag2idx)
        ).to(args.device)
    # print(model)
    # model = nn.DataParallel(model)
    
    trainer(train_loader, dev_loader, model, args, weights, data[TRAIN], data[DEV])

def trainer(train_loader, valid_loader, model, args, weights, train_set, val_set):

    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr) 
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)
    # writer = SummaryWriter() # Writer of tensoboard.

    if not os.path.isdir(args.ckpt_dir):
        os.mkdir(args.ckpt_dir) # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = args.num_epoch, math.inf, 0, 0

    best_acc = 0.0
    for epoch in range(n_epochs):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        model.train() # Set your model to train mode.

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y, lengths in train_pbar:
            optimizer.zero_grad()               # Set gradient to zero.
            x, y = x.to(args.device), y.to(args.device)   # Move your data to device. 
            pred = model(x, lengths)
            # print('x =', x.shape)
            # print('pred =', pred.shape)
            # print('y =', y.shape)
            loss = criterion(pred, y)
            loss.backward()                     # Compute gradient(backpropagation).
            optimizer.step()                    # Update parameters.
            step += 1
            
            _, train_pred = torch.max(pred, 1) # get the index of the class with the highest probability
            # print('train_pred =', train_pred.shape)
            # print('y =', y.shape)
            for i in range(len(y)):
                gt_vec = y[i][:lengths[i]]
                pred_vec = train_pred[i][:lengths[i]]
                # print('gt_vec =', gt_vec.shape)
                # print('pred_vec =', pred_vec.shape)
                acc = accuracy_score(gt_vec, pred_vec)
                if acc.item() == 1:
                    train_acc += 1
            train_loss += loss.item()

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        # writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval() # Set your model to evaluation mode.
        
        with torch.no_grad():
            for x, y, lengths in valid_loader:
                x, y = x.to(args.device), y.to(args.device)
            
                pred = model(x, lengths)
                loss = criterion(pred, y)

                _, val_pred = torch.max(pred, 1)
                # print(val_pred)
                # print(y)
                for i in range(len(y)):
                    gt_vec = y[i][:lengths[i]]
                    pred_vec = val_pred[i][:lengths[i]]
                    # print(gt_vec)
                    # print(pred_vec)
                    acc = accuracy_score(gt_vec, pred_vec)
                    if acc.item() == 1:
                        val_acc += 1
                val_loss += loss.item()
            
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
            epoch + 1, n_epochs, train_acc/len(train_set), train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(valid_loader)
        ))
        # writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.ckpt_dir / "model.ckpt") # Save your best model
            print('saving model with acc {:.3f}'.format(best_acc/len(val_set)))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= args.early_stop:
            print('\nModel is not improving, so we halt the training session.')
            return

        # scheduler.step()

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--seed", type=int, default=1121326)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    # parser.add_argument("--momentum", type=float, default=0.99)
    # parser.add_argument("--weight_decay", type=float, default=1e-5)

    # scheduler
    parser.add_argument("--step_size", type=float, default=30)

    # data loader
    parser.add_argument("--batch_size", type=int, default=64)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--early_stop", type=float, default=20)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    if torch.cuda.is_available():
        print('cuda is available')
    else:
        print('cuda is not available')
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
