CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --dataset_root OTTers/data \
    --domain out_of_domain \
    --output_dir tmp/out_of_domain \
    --max_epoch 10 \
    --train_bsize 16 \
    --eval_bsize 16 \
    --patience 3