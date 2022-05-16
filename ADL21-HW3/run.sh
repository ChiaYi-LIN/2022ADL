CUDA_VISIBLE_DEVICES=0,1 python run_summarization.py \
    --model_name_or_path ./tmp/mt5_small \
    --output_dir ./predictions \
    --overwrite_output_dir True \
    --do_predict \
    --test_file ${1} \
    --source_prefix "summarize: " \
    --text_column maintext \
    --num_beams 5 \
    --predict_with_generate

python postprocess.py --input ${1} --output ${2}