mkdir -p ./r10922124
cp ./intent_cls.sh ./r10922124/
cp ./slot_tag.sh ./r10922124/
cp ./README.md ./r10922124/
# cp ./report.pdf ./r10922124/
# cp ./download.sh ./r10922124/
cp ./utils.py ./r10922124/
cp ./dataset.py ./r10922124/
cp ./model.py ./r10922124/
cp ./train_intent.py ./r10922124/
cp ./test_intent.py ./r10922124/
cp ./train_slot.py ./r10922124/
cp ./test_slot.py ./r10922124/

zip adl_hw1_r10922124.zip ./cache/intent/* ./cache/slot/* ./data/intent/* ./data/slot/* ./ckpt/intent/model.ckpt ./ckpt/slot/model.ckpt