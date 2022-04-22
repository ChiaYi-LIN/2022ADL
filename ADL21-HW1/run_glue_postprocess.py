#%%
import csv

prediction_file = "./tmp/test-glue/predict_results_None.txt"
output_file = "./run_glue_predictions.csv"

#%%
with open(prediction_file, 'r') as in_file:
    predictions = [line.strip() for line in in_file]
    with open(output_file, 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('id', 'intent'))
        for i, pred in enumerate(predictions[1:]):
            writer.writerow((f'test-{i}', pred.split('\t')[1]))

#%%
