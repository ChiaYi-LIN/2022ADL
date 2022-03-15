# Create Directory
mkdir -p ./r10922124

# Add Mandatory Files
cp ./intent_cls.sh ./r10922124/
cp ./slot_tag.sh ./r10922124/
cp ./README.md ./r10922124/
cp ./report.pdf ./r10922124/
cp ./download.sh ./r10922124/

# For Environment
cp ./Makefile ./r10922124/
cp ./environment.yml ./r10922124/
cp ./requirements.in ./r10922124/
cp ./requirements.txt ./r10922124/

# For Training & Testing (Q2 & Q3)
cp ./utils.py ./r10922124/
cp ./dataset.py ./r10922124/
cp ./model.py ./r10922124/
cp ./train_intent.py ./r10922124/
cp ./test_intent.py ./r10922124/
cp ./train_slot.py ./r10922124/
cp ./test_slot.py ./r10922124/

# For Q4
cp ./eval_slot.py ./r10922124/

# For Q5
cp ./intent_Q2.py ./r10922124/
cp ./model_Q2.py ./r10922124/
cp ./intent_Q5.py ./r10922124/
cp ./model_Q5.py ./r10922124/

# For Download
rm -f adl_hw1_r10922124.zip
zip adl_hw1_r10922124.zip ./cache/intent/* ./cache/slot/* ./data/intent/* ./data/slot/* ./ckpt/intent/model.ckpt ./ckpt/slot/model.ckpt

# For NTUCool Submit 
rm -f r10922124_submit_hw1.zip
zip -r r10922124_submit_hw1.zip ./r10922124

# Clean Directory
rm -r ./r10922124