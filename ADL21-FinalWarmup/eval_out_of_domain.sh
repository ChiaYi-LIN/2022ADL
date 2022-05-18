python postprecess.py out_of_domain
sacrebleu ./OTTers/data/out_of_domain/test/target.txt  -i ./tmp/out_of_domain/generated_predictions.txt -m bleu -b -w 3 --lowercase
CUDA_VISIBLE_DEVICES=0,1 python perplexity.py ./tmp/out_of_domain/postprocessed_predictions.txt