python3 run_ner.py \
  --model_name_or_path bert-base-uncased \
  --train_file ./cache/slot/train_dict.json \
  --validation_file ./cache/slot/valid_dict.json \
  --test_file ./cache/slot/test_dict.json \
  --output_dir ./tmp/test-ner \
  --overwrite_output_dir True \
  --per_device_train_batch_size 16 \
  --do_train \
  --do_eval \
  --do_predict