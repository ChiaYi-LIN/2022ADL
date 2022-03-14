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
from seqeval.metrics import accuracy_score, classification_report
from seqeval.scheme import IOB2

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
    
    # Load Dataset
    datasets: Dict[str, SeqIOBDataset] = {
        split: SeqIOBDataset(x=split_data[['tokens']], y=split_data[['tags']], vocab=vocab, label_mapping=tag2idx, max_len=args.max_len)
        for split, split_data in data.items()
    }
    # print(datasets['train'].x)
    # print(datasets['train'].y)

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

    # load weights into model
    model.load_state_dict(torch.load(args.ckpt_path))

    eval_val(dev_loader, model, datasets[DEV])

def eval_val(valid_loader, model, val_set):
    all_gt, all_pred = [], []
    ttl_joint_acc = 0
    ttl_token_acc, token_cnt = 0, 0
    with torch.no_grad():
        for x, y, lengths in valid_loader:
            x, y = x.to(args.device), y.to(args.device)
            pred = model(x, lengths)
            _, val_pred = torch.max(pred, 1)
            # print(val_pred)
            # print(y)
            for i in range(len(y)):
                gt_vec = y[i][:lengths[i]]
                pred_vec = val_pred[i][:lengths[i]]
                # print(gt_vec)
                # print(pred_vec)
                
                # Joint Accuracy
                joint_acc = accuracy_score(gt_vec, pred_vec)
                if joint_acc.item() == 1:
                    ttl_joint_acc += 1

                # Token Accuracy
                ttl_token_acc += (pred_vec == gt_vec).sum().item()
                token_cnt += len(gt_vec)

                # Classification Report
                gt_label_vec = []
                pred_label_vec = []
                for idx in gt_vec.tolist():
                    gt_label_vec.append(val_set.idx2label(idx))
                for idx in pred_vec.tolist():
                    pred_label_vec.append(val_set.idx2label(idx))
                all_gt.append(gt_label_vec)
                all_pred.append(pred_label_vec)
        
        print('num gt =', len(all_gt))
        print('num pred =', len(all_pred))
        print('Joint Accuracy: {:.3f} ( {} / {} )' .format(ttl_joint_acc / len(all_gt), ttl_joint_acc, len(all_gt)))
        print('Token Accuracy: {:.3f} ( {} / {} )' .format(ttl_token_acc / token_cnt, ttl_token_acc, token_cnt))
        print('Classification Report:')
        print(classification_report(all_gt, all_pred, mode='strict', scheme=IOB2))
            

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
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/slot/model.ckpt",
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
