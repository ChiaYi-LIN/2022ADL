CUDA_VISIBLE_DEVICES=1 python run_glue.py \
  --model_name_or_path bert-base-cased \
  --train_file ./data/intent/train_dict.json \
  --validation_file ./data/intent/valid_dict.json \
  --test_file ./data/intent/test_dict.json \
  --output_dir ./tmp/test-glue/ \
  --overwrite_output_dir True \
  --per_device_train_batch_size 32 \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 3 \
  --max_seq_length 128 \
  --learning_rate 2e-5 \
  