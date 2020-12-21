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
import json
from itertools import chain

import utils
from models import StyleEncoder, StyleDecoder


def div_from_prior(posterior):
    prior = Normal(torch.zeros_like(posterior.loc), torch.ones_like(posterior.scale))
    return kl_divergence(posterior, prior).sum(dim=-1)


def evaluate_tf(update_step):
    encoder.eval()
    decoder.eval()
    ce_loss_all = []
    kl_loss_all = []
    loss_all = []

    for batch_raw in data_val:
        title_pos = batch_raw['title_pos']
        with torch.no_grad():
            styenc_dist = encoder(title_pos)
            styenc_sampled = styenc_dist.rsample()
            ce_loss = decoder(title_pos, styenc_sampled).item()
            kl_loss = div_from_prior(styenc_dist).mean().item()

        ce_loss_all.append(ce_loss)
        kl_loss_all.append(kl_loss)
        loss_all.append(ce_loss + kl_loss * args.kl_weight)

    mlflow.log_metric('ce_loss_eval', np.mean(ce_loss_all), update_step)
    mlflow.log_metric('kl_loss_eval', np.mean(kl_loss_all), update_step)
    mlflow.log_metric('loss_eval', np.mean(loss_all), update_step)


def train():
    update_step = 0
    timekeeper = utils.TimeKeeper(args.num_epochs)

    for epoch in range(1, args.num_epochs+1):
        for batch_raw in data_train:
            update_step += 1
            encoder.train()
            decoder.train()

            title_pos = batch_raw['title_pos']
            styenc_dist = encoder(title_pos)
            styenc_sampled = styenc_dist.rsample()
            ce_loss = decoder(title_pos, styenc_sampled)
            kl_loss = div_from_prior(styenc_dist).mean()

            loss = ce_loss + kl_loss * args.kl_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if update_step % args.log_freq == 0:
                mlflow.log_metric('ce_loss_train', ce_loss.item(), update_step)
                mlflow.log_metric('kl_loss_train', kl_loss.item(), update_step)
                mlflow.log_metric('loss_train', loss.item(), update_step)
                styenc_loc = styenc_dist.loc.abs().mean().item()
                styenc_scale = styenc_dist.scale.mean().item()
                mlflow.log_metric('styenc_loc', styenc_loc, update_step)
                mlflow.log_metric('styenc_scale', styenc_scale, update_step)

            if update_step % args.eval_freq == 0:
                evaluate_tf(update_step)

        eta_hour, eta_min, eta_sec = timekeeper.get_eta(epoch)
        print(f"Epoch {epoch} done. ETA: {eta_hour:02d}:{eta_min:02d}:{eta_sec:02d}", flush=True)
        mlflow.log_metric('epoch', epoch)

    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / 'checkpoint'
        path.mkdir(exist_ok=True)

        with open(path / 'config.json', 'w') as f:
            json.dump({
                'embedding_dim': args.embedding_dim,
                'code_dim': args.code_dim,
                'hidden_dim': args.hidden_dim,
            }, f, indent=4)

        torch.save(encoder.state_dict(), path / 'encoder.pt')
        torch.save(decoder.state_dict(), path / 'decoder.pt')
        mlflow.log_artifact(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--code_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--kl_weight', type=float, default=1e-3)
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--eval_freq', type=int, default=1000)
    args = parser.parse_args()

    with open('data/preprocessed.pkl', 'rb') as f:
        data = pickle.load(f)

    data_train, data_val, _ = utils.split_data(data)
    data_train = utils.arXivDataLoader(data_train, args.batch_size)
    data_val = utils.arXivDataLoader(data_val, args.batch_size)

    print("data loaded", flush=True)

    device = 'cuda'
    encoder = StyleEncoder(
        embedding_dim=args.embedding_dim,
        code_dim=args.code_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    decoder = StyleDecoder(
        encoder.embedding,
        code_dim=args.code_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = Adafactor(
        chain(encoder.parameters(), decoder.parameters()),
        lr=args.lr,
        relative_step=False,
        scale_parameter=False
    )

    print("model loaded", flush=True)

    mlflow.set_experiment(args.exp_name)
    run = mlflow.start_run()
    mlflow.log_params({
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'embedding_dim': args.embedding_dim,
        'code_dim': args.code_dim,
        'hidden_dim': args.hidden_dim,
        'lr': args.lr,
        'kl_weight': args.kl_weight,
    })

    train()
