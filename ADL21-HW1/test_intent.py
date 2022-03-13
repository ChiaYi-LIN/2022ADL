import csv
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

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = pd.read_json(args.test_file)
    dataset = SeqClsDataset(x=data[['text']], vocab=vocab, label_mapping=intent2idx, max_len=args.max_len)

    # TODO: crecate DataLoader for test dataset
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings=embeddings,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        num_class=len(intent2idx)
        ).to(args.device)
    # model = nn.DataParallel(model)

    # load weights into model
    model.load_state_dict(torch.load(args.ckpt_path))

    # TODO: predict dataset
    preds = predict(test_loader, model, args.device) 

    # TODO: write prediction to file (args.pred_file)
    save_pred(preds, dataset, args.pred_file)  

def predict(test_loader, model, device):
    model.eval() # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)                        
        with torch.no_grad():                   
            pred = model(x, [x.shape[1]])
            # print('x =', x.shape)
            # print('pred =', pred.shape)      
            # print(sum(pred[0]))
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()
    return preds

def save_pred(preds, dataset, file):
    ''' Save predictions to specified file '''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'intent'])
        for i, p in enumerate(preds):
            writer.writerow(['test-'+str(i), dataset.idx2label(np.argmax(p))])

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred_intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
