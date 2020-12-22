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
from pathlib import Path
import tempfile

from models import arXivModel
import utils


def evaluate_gen(num_samples=20):
    metric = load_metric("rouge", experiment_id=run.info.run_id)
    samples = []

    for batch_raw in data_val:
        generated = model.generate(batch_raw, num_beams=args.num_beams)
        metric.add_batch(predictions=generated, references=batch_raw['title'])

        if len(samples) < num_samples:
            for i in range(len(batch_raw['abstract'])):
                samples.append({
                    'abstract': batch_raw['abstract'][i],
                    'title_actual': batch_raw['title'][i],
                    'title_generated': generated[i],
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
        path = Path(tempdir, f'samples.txt')
        with open(path, 'w') as f:
            for sample, score in zip(samples, scores_f1):
                f.write(f"<Abstract>\n{utils.wrap_text(sample['abstract'], 80)}\n")
                f.write(f"<Actual title>\n{utils.wrap_text(sample['title_actual'], 80)}\n")
                f.write(f"<Generated title>\n{utils.wrap_text(sample['title_generated'], 80)}\n")
                f.write(f"Score: {score[0]:.2f}/{score[1]:.2f}/{score[2]:.2f}\n\n")

        mlflow.log_artifact(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_beams', type=int, default=4)
    args = parser.parse_args()

    device = 'cuda'

    with open('data/preprocessed.pkl', 'rb') as f:
        data = pickle.load(f)

    _, data_val, _ = utils.split_data(data)
    data_val = utils.arXivDataLoader(data_val, args.batch_size)
    print("data loaded", flush=True)

    model = arXivModel.from_checkpoint(args.checkpoint_path, device='cuda')
    print("model loaded", flush=True)

    mlflow.set_experiment(args.exp_name)
    run = mlflow.start_run()
    mlflow.log_params({
        'checkpoint_path': args.checkpoint_path,
        'batch_size': args.batch_size,
        'num_beams': args.num_beams,
    })

    evaluate_gen()
