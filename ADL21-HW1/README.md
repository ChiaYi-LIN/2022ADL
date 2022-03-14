## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl"
make
# otherwise
pip install -r requirements.txt
```

## Activate Environment
```shell
conda activate adl
```

<!-- ## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
``` -->

## Intent detection (training)
```shell
python train_intent.py
```

## Intent detection (predicting)
```shell
bash ./intent_cls.sh /path/to/test.json /path/to/pred.csv
```

## Slot tagging (training)
```shell
python train_slot.py
```

## Slot tagging (predicting)
```shell
bash ./slot_tag.sh /path/to/test.json /path/to/pred.csv
```

## Reproduce for Questions

### Q2
```shell
python train_intent.py
```

### Q3
```shell
python train_slot.py
```

### Q4
```shell
python eval_slot.py
```

### Q5
```shell
python intent_Q2.py
python intent_Q5.py
```