# For Code Submission
rm -rf ./r10922124
mkdir -p ./r10922124

cp ./train.sh ./r10922124
cp ./run_summarization.py ./r10922124
cp -r ./tw_rouge ./r10922124
cp ./download.sh ./r10922124
cp ./run.sh ./r10922124
cp ./postprocess.py ./r10922124
cp ./generate.py ./r10922124
cp ./README.md ./r10922124
cp ./report.pdf ./r10922124

rm -f r10922124.zip
zip -r r10922124.zip ./r10922124

rm -r ./r10922124

# For Dropbox
rm -rf ./adl_hw3_r10922124
mkdir -p ./adl_hw3_r10922124

cp -r ./data ./adl_hw3_r10922124
cp --parents ./tmp/mt5_small/* ./adl_hw3_r10922124

rm -f adl_hw3_r10922124.zip
zip -r adl_hw3_r10922124.zip ./adl_hw3_r10922124

rm -r ./adl_hw3_r10922124