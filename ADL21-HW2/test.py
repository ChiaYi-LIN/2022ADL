#%%
"""## Import Packages"""
import json
import numpy as np
import pandas as pd
from datasets import Dataset
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForMultipleChoice, 
    AutoModelForQuestionAnswering, 
    TrainingArguments, 
    Trainer, 
)

#%%
from argparse import ArgumentParser
from pathlib import Path
parser = ArgumentParser()
parser.add_argument(
    "--context_file",
    type=Path,
    help="Path to the context file.",
    required=True
)
parser.add_argument(
    "--test_file",
    type=Path,
    help="Path to the test file.",
    required=True
)
parser.add_argument(
    "--pred_file",
    type=Path,
    help="Path to the prediction file.",
    required=True
)
args = parser.parse_args()

#%%
output_name = "roberta_wwm_dup"
# bert-base-chinese
# hfl/chinese-roberta-wwm-ext
tokenizer_checkpoint = "hfl/chinese-roberta-wwm-ext"
CS_model_checkpoint = "./model/roberta_wwm/CS"
QA_model_checkpoint = "./model/roberta_wwm/QA"
max_length = 512 # 384 -> 512
stride = 128
device = "cuda" if torch.cuda.is_available() else "cpu"

#%%
"""## Read Data"""
def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data

context = read_data(args.context_file)
test_questions = read_data(args.test_file)
test_questions[0]

#%%
"""## Paragraph ID to Context"""
def unfold_questions(mode, questions):
    for question in questions:
        paragraphs = question['paragraphs']

        if mode != 'test':
            relevant = question['relevant']
            # label key
            label = paragraphs.index(relevant)
            question['labels'] = label

        # context key
        for i in range(4):
            paragraph_id = paragraphs[i]
            paragraph = context[paragraph_id]
            question[f'context{i}'] = paragraph

    return questions

unfold_test_questions = unfold_questions('test', test_questions)

#%%
"""## Data to Dataset"""
test_dataset = Dataset.from_pandas(pd.DataFrame(data=unfold_test_questions))

#%%
"""## Load Tokenizer"""
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

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
        padding="max_length", 
        truncation="only_second",
        max_length=max_length,
        return_tensors="pt",
    )
    output = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    return output

tokenized_test_dataset = test_dataset.map(CS_preprocess_function, batched=True, remove_columns=['id', 'question', 'context0', 'context1', 'context2', 'context3', 'paragraphs'])

#%%
"""## Load Model"""
from transformers import AutoModelForMultipleChoice
model = AutoModelForMultipleChoice.from_pretrained(CS_model_checkpoint).to(device)

#%%
"""## Trainer"""
from transformers import Trainer

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
)

#%%
"""## Make Predictions"""
predictions = trainer.predict(tokenized_test_dataset)
predict_labels = np.argmax(predictions[0], axis=1)

#%%
"""## Add Context Selection into Test Data"""
def get_relevant(x):
    paragraphs = x['paragraphs']
    label = int(x['labels'])
    return paragraphs[label]
CS_test_dataset = pd.DataFrame(test_dataset)
CS_test_dataset['labels'] = predict_labels
CS_test_dataset['relevant'] = CS_test_dataset.apply(get_relevant, axis=1)

#%%
"""## Prepare Dataset for QA"""
def get_context(x):
    label = x['labels']
    context = x[f'context{label}']
    return context

QA_test_dataset = pd.DataFrame(CS_test_dataset)
QA_test_dataset["context"] = QA_test_dataset.apply(get_context, axis=1)

#%%
"""## Keep Needed Columns Only"""
QA_test_dataset = QA_test_dataset[['id', 'question', 'context']]

#%%
"""## QA Data To Dataset"""
QA_test_dataset = Dataset.from_pandas(pd.DataFrame(data=QA_test_dataset))

#%%
"""## Load Tokenizer"""
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

#%%
"""## Load QA Model"""
from transformers import AutoModelForQuestionAnswering
model = AutoModelForQuestionAnswering.from_pretrained(QA_model_checkpoint).to(device)

#%%
"""## Post-processing & Compute Exact-Match"""
from transformers import EvalPrediction
from utils_qa import postprocess_qa_predictions
from datasets import load_metric

n_best = 20
max_answer_length = 30
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
    try:
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)
    except:
        return formatted_predictions

def compute_metrics(p: EvalPrediction):
    return metric.compute(predictions=p.predictions, references=p.label_ids)

#%%
"""## Trainer"""
from trainer_qa import QuestionAnsweringTrainer

training_args = TrainingArguments(
    output_dir=f"./results/{output_name}/QA",
)

trainer = QuestionAnsweringTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    post_process_function=post_processing_function,
    compute_metrics=compute_metrics,
)

#%%
"""## Preprocess Test Data"""
def QA_preprocess_test_examples(examples):
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

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

tokenized_QA_test_dataset = QA_test_dataset.map(QA_preprocess_test_examples, batched=True, remove_columns=QA_test_dataset.column_names)

#%%
"""## Generate Predictions"""
predictions = trainer.predict(tokenized_QA_test_dataset, QA_test_dataset)
predictions_df = pd.DataFrame(predictions)
predictions_df['answer'] = predictions_df['prediction_text']
predictions_df[['id', 'answer']].to_csv(args.pred_file, index=False)

#%%
