#%%
"""## Import Packages"""
import json
import numpy as np
import pandas as pd
import random
from datasets import Dataset
import torch
from torch.utils.data import DataLoader 
from transformers import AdamW, BertForMultipleChoice, BertTokenizerFast

from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# Fix random seed for reproducibility
set_seed = 0
def same_seeds(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
same_seeds(set_seed)

#%%
"""## Read Data"""
def read_data(file):
	with open(file, 'r', encoding="utf-8") as reader:
		data = json.load(reader)
	return data

context = read_data("./data/context.json")
train_questions = read_data("./data/train.json")
valid_questions = read_data("./data/valid.json")
test_questions = read_data("./data/test.json")
train_questions[0]

#%%
"""## Paragraph ID to Context"""
def unfold_questions(mode, questions):
	for question in questions:
		paragraphs = question['paragraphs']

		if mode != 'test':
			relevant = question['relevant']
			# label key
			label = paragraphs.index(relevant)
			question['label'] = label

		# context key
		for i in range(4):
			paragraph_id = paragraphs[i]
			paragraph = context[paragraph_id]
			question[f'context{i}'] = paragraph

	return questions

unfold_train_questions = unfold_questions('train', train_questions)
unfold_valid_questions = unfold_questions('valid', valid_questions)
unfold_test_questions = unfold_questions('test', test_questions)

#%%
"""## Data to Dataset"""
train_dataset = Dataset.from_pandas(pd.DataFrame(data=unfold_train_questions))
valid_dataset = Dataset.from_pandas(pd.DataFrame(data=unfold_valid_questions))
test_dataset = Dataset.from_pandas(pd.DataFrame(data=unfold_test_questions))

#%%
"""## Load Tokenizer"""
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

#%%
"""## Preprocess & Tokenize Data"""
def preprocess_function(examples):
	first_sentences = [[question] * 4 for question in examples["question"]]
	ending_names = ["context0", "context1", "context2", "context3"]
	second_sentences = [
		[f"{examples[end][i]}" for end in ending_names] for i, _ in enumerate(examples["question"])
	]
	first_sentences = sum(first_sentences, [])
	second_sentences = sum(second_sentences, [])
	
	tokenized_examples = tokenizer(first_sentences, second_sentences, padding=True, truncation=True, return_tensors="pt")
	output = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
	return output

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

#%%
for k, v in tokenized_train_dataset[0].items() :
    print (k)

#%%
# from dataclasses import dataclass
# from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
# from typing import Optional, Union
# import torch

# @dataclass
# class DataCollatorForMultipleChoice:
#     """
#     Data collator that will dynamically pad the inputs for multiple choice received.
#     """

#     tokenizer: PreTrainedTokenizerBase
#     padding: Union[bool, str, PaddingStrategy] = True
#     max_length: Optional[int] = None
#     pad_to_multiple_of: Optional[int] = None

#     def __call__(self, features):
#         label_name = "label" if "label" in features[0].keys() else "labels"
#         labels = [feature.pop(label_name) for feature in features]
#         batch_size = len(features)
#         num_choices = len(features[0]["input_ids"])
#         flattened_features = [
#             [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
#         ]
#         flattened_features = sum(flattened_features, [])

#         batch = self.tokenizer.pad(
#             flattened_features,
#             padding=self.padding,
#             max_length=self.max_length,
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors="pt",
#         )

#         batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
#         batch["labels"] = torch.tensor(labels, dtype=torch.int64)
#         return batch

#%%
"""## Load Model"""
model = BertForMultipleChoice.from_pretrained("bert-base-chinese")

#%%
"""## Training"""
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    seed=set_seed,
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    # warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    label_names=["label"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,
    # tokenizer=tokenizer,
    # data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
)

trainer.train()

#%%
"""## Save Model"""
trainer.save_model('./model/context_select')

#%%
"""## Make Predictions"""
predictions = trainer.predict(tokenized_test_dataset)
predict_labels = np.argmax(predictions[0], axis=1)

#%%
"""## Add Context Selection into Test Data"""
def get_relevant(x):
    paragraphs = x['paragraphs']
    label = int(x['label'])
    return paragraphs[label]
CS_test_dataset = pd.DataFrame(test_dataset)
CS_test_dataset['label'] = predict_labels
CS_test_dataset['relevant'] = CS_test_dataset.apply(get_relevant, axis=1)

#%%
# import json
# with open('./data/cs_test.json', 'w') as fp:
#     json.dump(CS_test_dataset.to_dict('records'), fp, ensure_ascii=False)

#%%
"""## Prepare Dataset for QA"""
def get_context(x):
    label = x['label']
    context = x[f'context{label}']
    return context

QA_train_dataset = pd.DataFrame(train_dataset)
QA_valid_dataset = pd.DataFrame(valid_dataset)
QA_test_dataset = pd.DataFrame(CS_test_dataset)
QA_train_dataset["context"] = QA_train_dataset.apply(get_context, axis=1)
QA_valid_dataset["context"] = QA_valid_dataset.apply(get_context, axis=1)
QA_test_dataset["context"] = QA_test_dataset.apply(get_context, axis=1)

#%%
QA_train_dataset = QA_train_dataset[['id', 'question', 'context', 'answer']]
QA_valid_dataset = QA_valid_dataset[['id', 'question', 'context', 'answer']]
QA_test_dataset = QA_test_dataset[['id', 'question', 'context']]

#%%
"""## To Dataset"""
QA_train_dataset = Dataset.from_pandas(pd.DataFrame(data=QA_train_dataset))
QA_valid_dataset = Dataset.from_pandas(pd.DataFrame(data=QA_valid_dataset))
QA_test_dataset = Dataset.from_pandas(pd.DataFrame(data=QA_test_dataset))
QA_train_dataset[0]

#%%
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")



#%%
from transformers import BertForQuestionAnswering
model = BertForQuestionAnswering.from_pretrained("bert-base-chinese")

#%%
# """## Dataset and Dataloader"""
# class CS_Dataset(Dataset):
# 	def __init__(self, split, questions, tokenized_questions, tokenized_paragraphs):
# 		self.split = split # train, valid, test
# 		self.questions = questions
# 		self.tokenized_questions = tokenized_questions
# 		self.tokenized_paragraphs = tokenized_paragraphs
# 		self.max_question_len = 512
# 		self.max_paragraph_len = 512
		
# 		##### TODO: Change value of doc_stride #####
# 		self.doc_stride = 150

# 		# Input sequence length = [CLS] + question + [SEP] + paragraph + [SEP]
# 		self.max_seq_len = 1 + self.max_question_len + 1 + self.max_paragraph_len + 1

# 	def __len__(self):
# 		return len(self.questions)

# 	def __getitem__(self, idx):
# 		question = self.questions[idx]
# 		tokenized_question = self.tokenized_questions[idx]
# 		tokenized_paragraph = self.tokenized_paragraphs[question["paragraphs"]]

		