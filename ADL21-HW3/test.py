#%%
from tw_rouge import get_rouge

#%%
x = get_rouge(['我是人'], ['我是一個人'])

#%%
type(x)
#%%
result = {'eval_loss': 3.6235923767089844, 'eval_rouge-1': {'r': 0.19801484133530045, 'p': 0.2948330140312273, 'f': 0.22779721386295987}, 'eval_rouge-2': {'r': 0.07303326666729626, 'p': 0.10383914840367318, 'f': 0.08252850487355891}, 'eval_rouge-l': {'r': 0.1783484013901947, 'p': 0.2656305635168911, 'f': 0.20497953433853353}, 'eval_runtime': 179.3576, 'eval_samples_per_second': 30.632, 'eval_steps_per_second': 1.918, 'epoch': 3.87}
#%%
import pandas as pd
data = pd.read_csv('./data/train.csv')

#%%
data = data[['title', 'maintext']]

#%%
data.head()

#%%
from transformers import AutoTokenizer
model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

#%%
inputs = tokenizer("榜首進台大醫科卻休學、27歲拿到法國天文博士 李悅寧跌破眾人眼鏡返台任教")

#%%
tokenizer.convert_ids_to_tokens(inputs.input_ids)
#%%
