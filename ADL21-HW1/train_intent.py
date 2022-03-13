import os
import math
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

from dataset import SeqClsDataset
from utils import Vocab
from model import SeqClassifier

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

    # intent2idx.json is for intent2idx
    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())
    
    # read in train.json and eval.json
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: pd.read_json(path) for split, path in data_paths.items()}

    # Load Dataset
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(x=split_data[['text']], y=split_data[['intent']], vocab=vocab, label_mapping=intent2idx, max_len=args.max_len)
        for split, split_data in data.items()
    }
    # print(datasets['train'].x)

    # TODO: crecate DataLoader for train / dev datasets
    train_loader = DataLoader(datasets[TRAIN], batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=datasets[TRAIN].collate_fn)
    dev_loader = DataLoader(datasets[DEV], batch_size=args.batch_size, shuffle=False, pin_memory=True, collate_fn=datasets[DEV].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # print(embeddings.shape)

    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(
        embeddings=embeddings,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        num_class=len(intent2idx)
        ).to(args.device)
    # print(model)
    # model = nn.DataParallel(model)
    
    trainer(train_loader, dev_loader, model, args, data[TRAIN], data[DEV])

def trainer(train_loader, valid_loader, model, args, train_set, val_set):

    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.Adam(model.parameters()) 
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
            # print('pred =', pred.shape)       
            # print('y =', y.shape)       
            loss = criterion(pred, y)
            loss.backward()                     # Compute gradient(backpropagation).
            optimizer.step()                    # Update parameters.
            step += 1

            _, train_pred = torch.max(pred, 1) # get the index of the class with the highest probability
            train_acc += (train_pred.detach() == y.detach()).sum().item()
            train_loss += loss.item()
            
            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        # writer.add_scalar('Loss/train', train_loss/len(train_loader), step)

        model.eval() # Set your model to evaluation mode.
        with torch.no_grad():
            for x, y, lengths in valid_loader:
                x, y = x.to(args.device), y.to(args.device)
                
                pred = model(x, lengths)
                loss = criterion(pred, y)

                _, val_pred = torch.max(pred, 1) 
                val_acc += (val_pred.cpu() == y.cpu()).sum().item() # get the index of the class with the highest probability
                val_loss += loss.item()
            
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
            epoch + 1, n_epochs, train_acc/len(train_set), train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(valid_loader)
        ))
        
        # writer.add_scalar('Loss/valid', val_loss/len(valid_loader), step)

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

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--seed", type=int, default=1121326)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    # parser.add_argument("--lr", type=float, default=1e-3)
    # parser.add_argument("--momentum", type=float, default=0.99)
    # parser.add_argument("--weight_decay", type=float, default=1e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=16)

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
