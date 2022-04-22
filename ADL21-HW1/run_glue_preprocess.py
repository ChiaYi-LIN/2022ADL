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
                        "text" : row["text"],
                        "label" : "book_flight",
                    }, ensure_ascii=False)
                )
            else:
                writer.write(
                    json.dumps({
                        "text" : row["text"],
                        "label" : row["intent"],
                    }, ensure_ascii=False)
                )
            writer.write("\n")

#%%
train_data = read_json('./data/intent/train.json')
valid_data = read_json('./data/intent/eval.json')
test_data = read_json('./data/intent/test.json')

#%%
write_json("./data/intent/train_dict.json", train_data, "train")
write_json("./data/intent/valid_dict.json", valid_data, "valid")
write_json("./data/intent/test_dict.json", test_data, "test")

#%%
