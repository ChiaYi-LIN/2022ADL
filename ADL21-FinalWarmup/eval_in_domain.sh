python postprecess.py in_domain
sacrebleu ./OTTers/data/in_domain/test/target.txt  -i ./tmp/in_domain/generated_predictions.txt -m bleu -b -w 3 --lowercase
CUDA_VISIBLE_DEVICES=0,1 python perplexity.py ./tmp/in_domain/postprocessed_predictions.txt