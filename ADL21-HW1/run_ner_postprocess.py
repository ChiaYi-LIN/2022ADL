#%%
import csv

prediction_file = "./tmp/test-ner/predictions.txt"
output_file = "./run_ner_predictions.csv"

#%%
with open(prediction_file, 'r') as in_file:
    predictions = [line.strip() for line in in_file]
    with open(output_file, 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('id', 'tags'))
        for i, pred in enumerate(predictions):
            writer.writerow((f'test-{i}', pred))

#%%
