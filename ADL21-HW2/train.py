#%%
"""## Import Packages"""
import os
import random
import json
import numpy as np
import pandas as pd
from datasets import Dataset
import torch
from transformers import (
    BertTokenizerFast, 
    BertForMultipleChoice, 
    BertForQuestionAnswering, 
    QuestionAnsweringPipeline,
    DefaultDataCollator,
    TrainingArguments, 
    Trainer, 
)

output_name = "bert"
# bert-base-chinese
# hfl/chinese-bert-wwm-ext
checkpoint = "bert-base-chinese"
batch_size = 2
num_epoch = 1
set_seed = 0
max_length = 384
stride = 50

#%%
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Fix random seed for reproducibility
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
"""
# Context Selection
"""

#%%
"""## Load Tokenizer"""
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained(checkpoint)

#%%
"""## Preprocess & Tokenize Data"""
def CS_preprocess_function(examples):
    first_sentences = [[question] * 4 for question in examples["question"]]
    ending_names = ["context0", "context1", "context2", "context3"]
    second_sentences = [
        [examples[end][i] for end in ending_names] for i, _ in enumerate(examples["question"])
    ]
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])
    
    tokenized_examples = tokenizer(
        first_sentences, 
        second_sentences, 
        padding=True, 
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    output = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    return output

tokenized_train_dataset = train_dataset.map(CS_preprocess_function, batched=True)
tokenized_valid_dataset = valid_dataset.map(CS_preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(CS_preprocess_function, batched=True)

#%%
# print(tokenized_train_dataset.features)
for k, v in tokenized_train_dataset[0].items():
    print (k)

#%%
"""## Load Data Collator"""
from transformers import DefaultDataCollator
data_collator = DefaultDataCollator()

#%%
"""## Load Model"""
from transformers import BertForMultipleChoice
model = BertForMultipleChoice.from_pretrained(checkpoint)

#%%
"""## Trainer"""
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    seed=set_seed,
    output_dir=f"./results/{output_name}/CS",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epoch,
    # warmup_steps=500,
    weight_decay=0.01,
    label_names=["label"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

#%%
"""## Training"""
trainer.train()

#%%
"""## Save Model"""
trainer.save_model(f'./model/{output_name}/CS')

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
"""## Save Context-Selected Test Data"""
import json
from pathlib import Path

Path(f"./data/{output_name}").mkdir(parents=True, exist_ok=True)
with open(f'./data/{output_name}/cs_test.json', 'w') as fp:
    json.dump(CS_test_dataset.to_dict('records'), fp, ensure_ascii=False)

#%%
"""
# Question Answering
"""
#%%
"""## Load Context-Selected Test Data"""
import json

with open(f'./data/{output_name}/cs_test.json') as f:
    CS_test_dataset = json.load(f)
CS_test_dataset[0]

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
"""## Squad answers format"""
def to_squad_answers_format(x):
    return {'answer_start': [x['answer']['start']], 'text':[x['answer']['text']]}

QA_train_dataset['answers'] = QA_train_dataset.apply(to_squad_answers_format, axis=1)
QA_valid_dataset['answers'] = QA_valid_dataset.apply(to_squad_answers_format, axis=1)

#%%
"""## Keep needed columns only"""
QA_train_dataset = QA_train_dataset[['id', 'question', 'context', 'answers']]
QA_valid_dataset = QA_valid_dataset[['id', 'question', 'context', 'answers']]
QA_test_dataset = QA_test_dataset[['id', 'question', 'context']]

#%%
"""## QA data To Dataset"""
QA_train_dataset = Dataset.from_pandas(pd.DataFrame(data=QA_train_dataset))
QA_valid_dataset = Dataset.from_pandas(pd.DataFrame(data=QA_valid_dataset))
QA_test_dataset = Dataset.from_pandas(pd.DataFrame(data=QA_test_dataset))
QA_train_dataset[0]

#%%
"""## Load Tokenizer"""
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained(checkpoint)

#%%
"""## Preprocess & Tokenize Data"""
def QA_preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs["offset_mapping"]
    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
        
    inputs["example_id"] = example_ids
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_QA_train_dataset = QA_train_dataset.map(QA_preprocess_function, batched=True, remove_columns=QA_train_dataset.column_names)
tokenized_QA_valid_dataset = QA_valid_dataset.map(QA_preprocess_function, batched=True, remove_columns=QA_valid_dataset.column_names)

#%%
for k, v in tokenized_QA_train_dataset[0].items() :
    print (k)

#%%
"""## Load Data Collator"""
from transformers import DefaultDataCollator
data_collator = DefaultDataCollator()

#%%
"""## Load QA Model"""
from transformers import BertForQuestionAnswering
model = BertForQuestionAnswering.from_pretrained(checkpoint)

#%%
# Post-processing:
from transformers import EvalPrediction
from utils_qa import postprocess_qa_predictions
from datasets import load_metric

n_best = 20
max_answer_length = 50
metric = load_metric("squad")

def post_processing_function(examples, features, predictions, stage="eval"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=False,
        n_best_size=n_best,
        max_answer_length=max_answer_length,
        output_dir=training_args.output_dir,
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)

def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

#%%
"""## Trainer"""
from transformers import TrainingArguments, Trainer
from trainer_qa import QuestionAnsweringTrainer

training_args = TrainingArguments(
    seed=set_seed,
    output_dir=f"./results/{output_name}/QA",
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epoch,
    # warmup_steps=500,
    weight_decay=0.01,
    # logging_dir=f"./logs/{output_name}/QA",
    logging_steps=1500,
    save_steps=1500,
    label_names=["start_positions", "end_positions"],
)

trainer = QuestionAnsweringTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_QA_train_dataset,
    eval_dataset=tokenized_QA_valid_dataset,
    eval_examples=QA_valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    post_process_function=post_processing_function,
    compute_metrics=compute_metrics,
)

#%%
"""## Training"""
trainer.train()

#%%
"""## Save QA Model"""
trainer.save_model(f'./model/{output_name}/QA')

#%%
"""## Load Answerer"""
from transformers import QuestionAnsweringPipeline
question_answerer = QuestionAnsweringPipeline(model=trainer.model, tokenizer=tokenizer, device=0)

#%%
"""## Answer questions"""
predictions = question_answerer(question=QA_test_dataset['question'], context=QA_test_dataset['context'])

#%%
"""## Generate output file"""
predictions_df = pd.DataFrame(predictions)
answer_test_dataset = pd.DataFrame(QA_test_dataset)
answer_test_dataset['answer'] = predictions_df['answer']
answer_test_dataset[['id', 'answer']].to_csv(f'./{output_name}.csv', index=False)

#%%
