#%%
import json
import pandas as pd

#%%
def read_json(filename):
    with open(filename, 'r', encoding='utf-8') as reader:
        return json.load(reader)

def write_json(filename, data):
    with open(filename, 'w', encoding='utf-8') as writer:
        for data in train_data:
            writer.write(
                json.dumps({
                    "id" : data["id"],
                    "ner_tags" : data["tags"],
                    "tokens" : data["tokens"],
                }, ensure_ascii=False)
            )
            writer.write("\n")

#%%
train_data = read_json('./data/slot/train.json')
valid_data = read_json('./data/slot/eval.json')
test_data = read_json('./data/slot/test.json')

#%%
write_json("./cache/slot/train_dict.json", train_data)
write_json("./cache/slot/valid_dict.json", valid_data)
write_json("./cache/slot/test_dict.json", test_data)

#%%
