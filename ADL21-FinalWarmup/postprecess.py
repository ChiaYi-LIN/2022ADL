#%%
import sys
import pandas as pd

#%%
domain = sys.argv[1]
# domain = 'in_domain'

#%%
texts = pd.read_csv(f"./OTTers/data/{domain}/test/text.csv")

#%%
preds = pd.read_csv(f"./tmp/{domain}/generated_predictions.txt", sep="\t", header=None)
preds.columns = ['prediction']

#%%
texts[['input_col', 'target_col']] = texts['inputs'].str.split('@', 1, expand=True)
texts['pred_col'] = preds['prediction']
texts['text'] = texts['input_col'] + texts['pred_col'] + texts['target_col'] 

#%%
texts[['text']].to_csv(f"./tmp/{domain}/postprocessed_predictions.txt")

#%%
