# To train the bert-base-chinese model
In train.py:
```python
tokenizer_checkpoint = "bert-base-chinese"
CS_model_checkpoint = "bert-base-chinese"
QA_model_checkpoint = "bert-base-chinese"
max_length = 384
```
and then:
```shell
python train.py
```

# To train the hfl/chinese-roberta-wwm-ext model (strong baseline) (plot learning curves)
In train.py:
```python
tokenizer_checkpoint = "hfl/chinese-roberta-wwm-ext"
CS_model_checkpoint = "hfl/chinese-roberta-wwm-ext"
QA_model_checkpoint = "hfl/chinese-roberta-wwm-ext"
max_length = 512
```
and then:
```shell
python train.py
```

# To train model without pretrain
```shell
python no_pretrain.py
```
