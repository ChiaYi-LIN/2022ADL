#%%
import pandas as pd
import json

#%%
def read_data(path):
    with open(path, 'r') as json_file:
        json_list = list(json_file)

    data = []
    for json_str in json_list:
        result = json.loads(json_str)
        data += [result]

    return pd.DataFrame(data)

#%%
train = read_data('./data/train.jsonl')
public = read_data('./data/public.jsonl')

#%%
train.to_csv('./data/train.csv')
public.to_csv('./data/public.csv')

#%%