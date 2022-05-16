#%%
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode
from transformers.utils.versions import require_version
from tw_rouge import get_rouge
import pandas as pd
import jsonlines
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    BeamSearchScorer,
)
import torch

#%%
prefix = "summarize: "
text_column = "maintext"
summary_column = "title"
id_column = "id"

#%%
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("./tmp/mt5_small").to('cuda')

#%%
titles, maintexts, ids = [], [], []
with jsonlines.open('./data/public.jsonl') as reader:
    for obj in reader:
        titles += [obj[summary_column]]
        maintexts += [obj[text_column]]
        ids += [obj[id_column]]

#%%
all_input_ids = []
for maintext in maintexts:
    all_input_ids += [tokenizer(maintext, return_tensors="pt").input_ids]

#%%
def generation_function(greedy=False, num_beams=1, top_k=50, top_p=1.0, temperature=1.0):
    do_sample = False if greedy else True
    
    results = []
    for input_ids in tqdm(all_input_ids):
        outputs = model.generate(
            input_ids.to('cuda'),
            do_sample=do_sample,
            max_length=64,
            num_beams=num_beams,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature
        )
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        results += [result[0]]

    return results

#%%
with open('./gen_results.txt', 'w') as f:
    f.write('')

#%%
print('Processing Greedy\n')
greedy_results = generation_function(greedy=True)
greedy_rouge = get_rouge(greedy_results, titles)

#%%
with open('./gen_results.txt', 'a') as f:
    f.write('Greedy:\n')
    for rouge_type in greedy_rouge:
        for sub_type in greedy_rouge[rouge_type]:
            f.write(f'{rouge_type}_{sub_type}: {greedy_rouge[rouge_type][sub_type]}\n')
    f.write('\n')
    
#%%
print('Processing No Greedy\n')
no_greedy_results = generation_function(greedy=False)
no_greedy_rouge = get_rouge(no_greedy_results, titles)

#%%
with open('./gen_results.txt', 'a') as f:
    f.write('No Greedy:\n')
    for rouge_type in no_greedy_rouge:
        for sub_type in no_greedy_rouge[rouge_type]:
            f.write(f'{rouge_type}_{sub_type}: {no_greedy_rouge[rouge_type][sub_type]}\n')
    f.write('\n')

#%%
print('Processing Beam Search 3\n')
beam_search_3_results = generation_function(greedy=False, num_beams=3)
beam_search_3_rouge = get_rouge(beam_search_3_results, titles)

#%%
with open('./gen_results.txt', 'a') as f:
    f.write('Beam Search 3:\n')
    for rouge_type in beam_search_3_rouge:
        for sub_type in beam_search_3_rouge[rouge_type]:
            f.write(f'{rouge_type}_{sub_type}: {beam_search_3_rouge[rouge_type][sub_type]}\n')
    f.write('\n')

#%%
print('Processing Beam Search 5\n')
beam_search_5_results = generation_function(greedy=False, num_beams=5)
beam_search_5_rouge = get_rouge(beam_search_5_results, titles)

#%%
with open('./gen_results.txt', 'a') as f:
    f.write('Beam Search 5:\n')
    for rouge_type in beam_search_5_rouge:
        for sub_type in beam_search_5_rouge[rouge_type]:
            f.write(f'{rouge_type}_{sub_type}: {beam_search_5_rouge[rouge_type][sub_type]}\n')
    f.write('\n')

#%%
print('Processing Top k 3\n')
top_k_3_results = generation_function(greedy=False, top_k=3)
top_k_3_rouge = get_rouge(top_k_3_results, titles)

#%%
with open('./gen_results.txt', 'a') as f:
    f.write('Top k 3:\n')
    for rouge_type in top_k_3_rouge:
        for sub_type in top_k_3_rouge[rouge_type]:
            f.write(f'{rouge_type}_{sub_type}: {top_k_3_rouge[rouge_type][sub_type]}\n')
    f.write('\n')

#%%
print('Processing Top k 5\n')
top_k_5_results = generation_function(greedy=False, top_k=5)
top_k_5_rouge = get_rouge(top_k_5_results, titles)

#%%
with open('./gen_results.txt', 'a') as f:
    f.write('Top k 5:\n')
    for rouge_type in top_k_5_rouge:
        for sub_type in top_k_5_rouge[rouge_type]:
            f.write(f'{rouge_type}_{sub_type}: {top_k_5_rouge[rouge_type][sub_type]}\n')
    f.write('\n')

#%%
print('Processing Top p 0.8\n')
top_p_8_results = generation_function(greedy=False, top_p=0.8)
top_p_8_rouge = get_rouge(top_p_8_results, titles)

#%%
with open('./gen_results.txt', 'a') as f:
    f.write('Top p 0.8:\n')
    for rouge_type in top_p_8_rouge:
        for sub_type in top_p_8_rouge[rouge_type]:
            f.write(f'{rouge_type}_{sub_type}: {top_p_8_rouge[rouge_type][sub_type]}\n')
    f.write('\n')

#%%
print('Processing Top p 0.9\n')
top_p_9_results = generation_function(greedy=False, top_p=0.9)
top_p_9_rouge = get_rouge(top_p_9_results, titles)

#%%
with open('./gen_results.txt', 'a') as f:
    f.write('Top p 0.9:\n')
    for rouge_type in top_p_9_rouge:
        for sub_type in top_p_9_rouge[rouge_type]:
            f.write(f'{rouge_type}_{sub_type}: {top_p_9_rouge[rouge_type][sub_type]}\n')
    f.write('\n')

#%%
print('Processing Temperature 0.8\n')
temperature_8_results = generation_function(greedy=False, temperature=0.8)
temperature_8_rouge = get_rouge(temperature_8_results, titles)

#%%
with open('./gen_results.txt', 'a') as f:
    f.write('Temperature 0.8:\n')
    for rouge_type in temperature_8_rouge:
        for sub_type in temperature_8_rouge[rouge_type]:
            f.write(f'{rouge_type}_{sub_type}: {temperature_8_rouge[rouge_type][sub_type]}\n')
    f.write('\n')

#%%
print('Processing Temperature 1.2\n')
temperature_12_results = generation_function(greedy=False, temperature=1.2)
temperature_12_rouge = get_rouge(temperature_12_results, titles)

#%%
with open('./gen_results.txt', 'a') as f:
    f.write('Temperature 1.2:\n')
    for rouge_type in temperature_12_rouge:
        for sub_type in temperature_12_rouge[rouge_type]:
            f.write(f'{rouge_type}_{sub_type}: {temperature_12_rouge[rouge_type][sub_type]}\n')
    f.write('\n')

#%%