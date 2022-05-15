import jsonlines
from pathlib import Path
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--preds",
    type=Path,
    default="./predictions/generated_predictions.txt",
)
parser.add_argument(
    "--input",
    type=Path,
)
parser.add_argument(
    "--output",
    type=Path,
)

args = parser.parse_args()

preds = args.preds.read_text().split('\n')
texts = jsonlines.open(args.input)
text_ids = [text['id'] for text in texts]

data = [{'title': title, 'id': id} for title, id in zip(preds, text_ids)]
with jsonlines.open(args.output, mode='w') as writer:
    writer.write_all(data)
