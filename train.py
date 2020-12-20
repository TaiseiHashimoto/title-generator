import argparse
import mlflow
import numpy as np
import torch
from torch.distributions import (
    Normal,
    kl_divergence,
)
from transformers import (
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    Adafactor,
)
from datasets import load_metric
import pickle
from pathlib import Path
import tempfile

import utils
from models import arXivModel


def div_from_prior(posterior):
    prior = Normal(torch.zeros_like(posterior.loc), torch.ones_like(posterior.scale))
    return kl_divergence(posterior, prior).sum(dim=-1)


def evaluate_tf(update_step):
    model.eval()
    ce_loss_all = []
    loss_all = []
    if args.encode_style:
        kl_loss_all = []

    for batch_raw in data_val:
        with torch.no_grad():
            output, style_encoder_dist, style_encoder_sampled = model(batch_raw)

        ce_loss = output.loss.item()
        ce_loss_all.append(ce_loss)
        if args.encode_style:
            kl_loss = div_from_prior(style_encoder_dist).mean().item()
            kl_loss_all.append(kl_loss)
            loss_all.append(ce_loss + kl_loss * args.kl_weight)
        else:
            loss_all.append(ce_loss)

    mlflow.log_metric('ce_loss_eval', np.mean(ce_loss_all), update_step)
    mlflow.log_metric('loss_eval', np.mean(loss_all), update_step)
    if args.encode_style:
        mlflow.log_metric('kl_loss_eval', np.mean(kl_loss_all), update_step)


def evaluate_gen(epoch, num_samples=20):
    model.eval()
    metric = load_metric("rouge")
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
    }, epoch)

    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / f'samples_{epoch}.txt'
        with open(path, 'w') as f:
            for sample, score in zip(samples, scores_f1):
                f.write(f"<Abstract>\n{utils.wrap_text(sample['abstract'], 80)}\n")
                f.write(f"<Actual title>\n{utils.wrap_text(sample['title_actual'], 80)}\n")
                f.write(f"<Generated title>\n{utils.wrap_text(sample['title_generated'], 80)}\n")
                f.write(f"Score: {score[0]:.2f}/{score[1]:.2f}/{score[2]:.2f}\n\n")

        mlflow.log_artifact(path)


def train():
    update_step = 0
    timekeeper = utils.TimeKeeper(args.num_epochs)

    for epoch in range(1, args.num_epochs+1):
        for batch_raw in data_train:
            update_step += 1

            model.train()
            output, style_encoder_dist, style_encoder_sampled = model(batch_raw)
            
            ce_loss = output.loss
            if args.encode_style:
                kl_loss = div_from_prior(style_encoder_dist).mean()
                loss = ce_loss + kl_loss * args.kl_weight
            else:
                loss = ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if update_step % args.log_freq == 0:
                mlflow.log_metric('ce_loss_train', ce_loss.item(), update_step)
                mlflow.log_metric('loss_train', loss.item(), update_step)
                if args.encode_style:
                    mlflow.log_metric('kl_loss_train', kl_loss.item(), update_step)
                    style_encoder_loc = style_encoder_dist.loc.abs().mean().item()
                    style_encoder_scale = style_encoder_dist.scale.mean().item()
                    mlflow.log_metric('style_encoder_loc', style_encoder_loc, update_step)
                    mlflow.log_metric('style_encoder_scale', style_encoder_scale, update_step)

            if update_step % args.eval_freq == 0:
                evaluate_tf(update_step)

        evaluate_gen(epoch)

        eta_hour, eta_min, eta_sec = timekeeper.get_eta(epoch)
        print(f"Epoch {epoch} done. ETA: {eta_hour:02d}:{eta_min:02d}:{eta_sec:02d}", flush=True)
        mlflow.log_metric('epoch', epoch)

    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / 'checkpoint'
        model.save(path)
        mlflow.log_artifact(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--model_lr', type=float, default=1e-5)
    parser.add_argument('--encode_style', action='store_true')
    parser.add_argument('--style_encoder_layer', type=int, default=3)
    parser.add_argument('--style_encoder_dim', type=int, default=32)
    parser.add_argument('--style_encoder_ffn_dim', type=int, default=128)
    parser.add_argument('--style_encoder_head', type=int, default=2)
    parser.add_argument('--style_encoder_lr', type=float, default=1e-3)
    parser.add_argument('--kl_weight', type=float, default=1.0)
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--eval_freq', type=int, default=1000)
    args = parser.parse_args()

    if not args.encode_style:
        args.style_encoder_layer = None
        args.style_encoder_dim = None
        args.style_encoder_ffn_dim = None
        args.style_encoder_head = None

    with open('data/preprocessed.pkl', 'rb') as f:
        data = pickle.load(f)

    data_train, data_val, _ = utils.split_data(data)
    data_train = utils.arXivDataLoader(data_train, args.batch_size)
    data_val = utils.arXivDataLoader(data_val, args.batch_size)

    print("data loaded", flush=True)

    model = arXivModel(
        model_name='google/pegasus-xsum',
        encode_style=args.encode_style,
        style_encoder_layer=args.style_encoder_layer,
        style_encoder_dim=args.style_encoder_dim,
        style_encoder_ffn_dim=args.style_encoder_ffn_dim,
        style_encoder_head=args.style_encoder_head,
        device='cuda',
    )

    params = [{'params': model.model.parameters(), 'lr': args.model_lr}]
    if args.encode_style:
        params.extend([
            {'params': model.style_encoder.parameters(), 'lr': args.style_encoder_lr},
            {'params': model.projector.parameters(), 'lr': args.style_encoder_lr},
        ])

    optimizer = Adafactor(params, relative_step=False, scale_parameter=False)

    print("model loaded", flush=True)

    mlflow.set_experiment(args.exp_name)
    mlflow.log_params({
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'num_beams': args.num_beams,
        'model_lr': args.model_lr,
        'encode_style': args.encode_style,
        'style_encoder_layer': args.style_encoder_layer,
        'style_encoder_dim': args.style_encoder_dim,
        'style_encoder_ffn_dim': args.style_encoder_ffn_dim,
        'style_encoder_head': args.style_encoder_head,
        'style_encoder_lr': args.style_encoder_lr,
        'kl_weight': args.kl_weight,
    })

    train()
