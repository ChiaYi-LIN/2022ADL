#%%
import json
import pandas as pd

#%%
def read_json(filename):
    with open(filename, 'r', encoding='utf-8') as reader:
        return json.load(reader)

def write_json(filename, data, split):
    with open(filename, 'w', encoding='utf-8') as writer:
        for row in data:
            if split == "test":
                writer.write(
                    json.dumps({
                        "id" : row["id"],
                        "ner_tags" : ["O" for _ in range(len(row["tokens"]))],
                        "tokens" : row["tokens"],
                    }, ensure_ascii=False)
                )
            else:
                writer.write(
                    json.dumps({
                        "id" : row["id"],
                        "ner_tags" : row["tags"],
                        "tokens" : row["tokens"],
                    }, ensure_ascii=False)
                )
            writer.write("\n")

#%%
train_data = read_json('./data/slot/train.json')
valid_data = read_json('./data/slot/eval.json')
test_data = read_json('./data/slot/test.json')

#%%
write_json("./data/slot/train_dict.json", train_data, "train")
write_json("./data/slot/valid_dict.json", valid_data, "valid")
write_json("./data/slot/test_dict.json", test_data, "test")

#%%
