# For Code Submission
rm -rf ./r10922124
mkdir -p ./r10922124

cp ./train.py ./r10922124
cp ./trainer_qa.py ./r10922124
cp ./utils_qa.py ./r10922124
cp ./download.sh ./r10922124
cp ./test.py ./r10922124
cp ./run.sh ./r10922124
cp ./report.pdf ./r10922124
cp ./no_pretrain.py ./r10922124
cp ./README.md ./r10922124

rm -f r10922124.zip
zip -r r10922124.zip ./r10922124

rm -r ./r10922124

# For Dropbox
rm -rf ./adl_hw2_r10922124
mkdir -p ./adl_hw2_r10922124

cp --parents ./data/* ./adl_hw2_r10922124
cp -r --parents ./model/roberta_wwm ./adl_hw2_r10922124

rm -f adl_hw2_r10922124.zip
zip -r adl_hw2_r10922124.zip ./adl_hw2_r10922124

rm -r ./adl_hw2_r10922124