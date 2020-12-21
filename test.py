import argparse
import mlflow
import torch
import numpy as np
from transformers import (
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    Adafactor,
)
from datasets import load_metric
import pickle
from tqdm import tqdm
from pathlib import Path
import tempfile

import utils


def evaluate_gen(num_samples=10):
    model.eval()
    metric = load_metric("rouge")
    samples = []

    # for batch_raw in tqdm(data_val, desc="[eval]"):
    for batch_raw in data_val:
        batch = tokenizer.prepare_seq2seq_batch(
            src_texts=batch_raw['abstract'],
            return_tensors="pt",
        ).to(device)
        output = model.generate(**batch, num_beams=args.num_beams, decoder_start_token_id=args.decoder_start_token_id)
        prediction = tokenizer.batch_decode(output, skip_special_tokens=True)

        metric.add_batch(predictions=prediction, references=batch_raw['title'])

        if len(samples) < num_samples:
            for i in range(len(batch_raw['abstract'])):
                samples.append({
                    'abstract': batch_raw['abstract'][i],
                    'title_actual': batch_raw['title'][i],
                    'title_predicted': prediction[i],
                })

    scores_all = metric.compute(use_agregator=False, rouge_types=['rouge1', 'rouge2', 'rouge3'])
    scores_f1 = []
    for i in range(len(scores_all['rouge1'])):
        scores_f1.append((
            scores_all['rouge1'][i].fmeasure,
            scores_all['rouge2'][i].fmeasure,
            scores_all['rouge3'][i].fmeasure,
        ))
    scores_f1 = np.array(scores_f1) * 100.

    mlflow.log_metrics({
        'rouge1': scores_f1[:, 0].mean(axis=0),
        'rouge2': scores_f1[:, 1].mean(axis=0),
        'rouge3': scores_f1[:, 2].mean(axis=0),
    })

    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / f'samples.txt'
        with open(path, 'w') as f:
            for sample, score in zip(samples, scores_f1):
                f.write(f"<Abstract>\n{utils.wrap_text(sample['abstract'], 80)}\n")
                f.write(f"<Actual title>\n{utils.wrap_text(sample['title_actual'], 80)}\n")
                f.write(f"<Predicted title>\n{utils.wrap_text(sample['title_predicted'], 80)}\n")
                f.write(f"Score: {score[0]:.2f}/{score[1]:.2f}/{score[2]:.2f}\n\n")

        mlflow.log_artifact(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--decoder_start_token_id', type=int, default=0)
    args = parser.parse_args()

    model_name = 'google/pegasus-xsum'
    device = torch.device('cuda')

    with open('data/preprocessed.pkl', 'rb') as f:
        data = pickle.load(f)

    _, data_val, _ = utils.split_data(data)
    data_val = utils.arXivDataLoader(data_val, args.batch_size)

    print("data loaded", flush=True)

    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    model.load_state_dict(torch.load('mlruns/1/28aafa9e5b88475385699bbe79f1bcb8/artifacts/model.pt'))
    model.to(device)

    print("model loaded", flush=True)

    mlflow.set_experiment(args.exp_name)
    mlflow.log_params({
        'batch_size': args.batch_size,
        'num_beams': args.num_beams,
        'decoder_start_token_id': args.decoder_start_token_id,
    })

    evaluate_gen()
