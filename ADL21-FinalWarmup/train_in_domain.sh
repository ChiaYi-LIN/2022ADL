CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --dataset_root OTTers/data \
    --domain in_domain \
    --output_dir tmp/in_domain \
    --max_epoch 10 \
    --train_bsize 16 \
    --eval_bsize 16 \
    --patience 3