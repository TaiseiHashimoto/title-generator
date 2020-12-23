import argparse
import mlflow
import numpy as np
import torch
from torch.distributions import (
    Normal,
    kl_divergence,
)
from transformers import Adafactor
from datasets import load_metric
import pickle
from pathlib import Path
import tempfile
from itertools import chain

from models import (
    arXivModel,
    Summarizer,
    StyleEncoder,
    StyleDecoder,
)
import utils


def div_from_prior(posterior):
    prior = Normal(torch.zeros_like(posterior.loc), torch.ones_like(posterior.scale))
    return kl_divergence(posterior, prior).sum(dim=-1)


def evaluate_pre(update_step):
    ce_loss_all = []
    kl_loss_all = []
    loss_all = []

    for batch_raw in data_val:
        batch_raw['title_pos']

        with torch.no_grad():
            ce_loss, style_code_dist, style_code = model.forward_pretrain(batch_raw)

        ce_loss = ce_loss.item()
        kl_loss = div_from_prior(style_code_dist).mean().item()
        ce_loss_all.append(ce_loss)
        kl_loss_all.append(kl_loss)
        loss_all.append(ce_loss + kl_loss * args.kl_weight)

    mlflow.log_metric('ce_loss_preeval', np.mean(ce_loss_all), update_step)
    mlflow.log_metric('kl_loss_preeval', np.mean(kl_loss_all), update_step)
    mlflow.log_metric('loss_preeval', np.mean(loss_all), update_step)


def evaluate_tf(update_step):
    ce_loss_all = []
    loss_all = []
    if args.encode_style:
        kl_loss_all = []

    for batch_raw in data_val:
        with torch.no_grad():
            ce_loss, style_code_dist, style_code = model.forward_train(batch_raw)

        ce_loss = ce_loss.item()
        ce_loss_all.append(ce_loss)
        if args.encode_style:
            kl_loss = div_from_prior(style_code_dist).mean().item()
            kl_loss_all.append(kl_loss)
            loss_all.append(ce_loss + kl_loss * args.kl_weight)
        else:
            loss_all.append(ce_loss)

    mlflow.log_metric('ce_loss_eval', np.mean(ce_loss_all), update_step)
    mlflow.log_metric('loss_eval', np.mean(loss_all), update_step)
    if args.encode_style:
        mlflow.log_metric('kl_loss_eval', np.mean(kl_loss_all), update_step)


def evaluate_gen(epoch, num_samples=20):
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
    }, epoch)

    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir, f'samples_{epoch}.txt')
        with open(path, 'w') as f:
            for sample, score in zip(samples, scores_f1):
                f.write(f"<Abstract>\n{utils.wrap_text(sample['abstract'], 80)}\n")
                f.write(f"<Actual title>\n{utils.wrap_text(sample['title_actual'], 80)}\n")
                f.write(f"<Generated title>\n{utils.wrap_text(sample['title_generated'], 80)}\n")
                f.write(f"Score: {score[0]:.2f}/{score[1]:.2f}/{score[2]:.2f}\n\n")

        mlflow.log_artifact(path)


def pretrain():
    params = chain(style_encoder.parameters(), style_decoder.parameters())
    optimizer = Adafactor(params, lr=args.styenc_lr_pre, relative_step=False, scale_parameter=False)

    update_step = 0
    timekeeper = utils.TimeKeeper(args.num_epochs)

    for epoch in range(1, args.num_epochs_pre+1):
        for batch_raw in data_train:
            update_step += 1

            ce_loss, style_code_dist, style_code = model.forward_pretrain(batch_raw)
            kl_loss = div_from_prior(style_code_dist).mean()
            loss = ce_loss + kl_loss * args.kl_weight_pre

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if update_step % args.log_freq == 0:
                mlflow.log_metric('ce_loss_pretrain', ce_loss.item(), update_step)
                mlflow.log_metric('loss_pretrain', loss.item(), update_step)
                mlflow.log_metric('kl_loss_pretrain', kl_loss.item(), update_step)
                styenc_loc = style_code_dist.loc.abs().mean().item()
                styenc_scale = style_code_dist.scale.mean().item()
                mlflow.log_metric('styenc_loc_pretrain', styenc_loc, update_step)
                mlflow.log_metric('styenc_scale_pretrain', styenc_scale, update_step)

            if update_step % args.eval_freq == 0:
                evaluate_pre(update_step)

        eta_hour, eta_min, eta_sec = timekeeper.get_eta(epoch)
        print(f"Epoch {epoch} done. ETA: {eta_hour:02d}:{eta_min:02d}:{eta_sec:02d}", flush=True)


def train():
    params = [{'params': summarizer.parameters()}]
    if style_encoder is not None:
        params.append({'params': style_encoder.parameters(), 'lr': args.styenc_lr})
    optimizer = Adafactor(params, lr=args.model_lr, relative_step=False, scale_parameter=False)

    update_step = 0
    timekeeper = utils.TimeKeeper(args.num_epochs)

    for epoch in range(1, args.num_epochs+1):
        for batch_raw in data_train:
            update_step += 1

            ce_loss, style_code_dist, style_code = model.forward_train(batch_raw)

            if args.encode_style:
                kl_loss = div_from_prior(style_code_dist).mean()
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
                    styenc_mean = style_code_dist.loc.abs().mean().item()
                    styenc_std = style_code_dist.scale.mean().item()
                    mlflow.log_metric('styenc_mean_train', styenc_mean, update_step)
                    mlflow.log_metric('styenc_std_train', styenc_std, update_step)
                    scales = style_encoder.scales.detach().cpu().numpy()
                    mlflow.log_metric('styenc_scale_src_train', scales[0], update_step)
                    mlflow.log_metric('styenc_scale_tgt_train', scales[1], update_step)

            if update_step % args.eval_freq == 0:
                evaluate_tf(update_step)

        evaluate_gen(epoch)

        eta_hour, eta_min, eta_sec = timekeeper.get_eta(epoch)
        print(f"Epoch {epoch} done. ETA: {eta_hour:02d}:{eta_min:02d}:{eta_sec:02d}", flush=True)
        mlflow.log_metric('epoch', epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--num_epochs_pre', type=int, default=2)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--model_lr', type=float, default=1e-5)
    parser.add_argument('--encode_style', action='store_true')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--styenc_embedding_dim', type=int, default=64)
    parser.add_argument('--styenc_code_dim', type=int, default=64)
    parser.add_argument('--styenc_num_layers', type=int, default=1)
    parser.add_argument('--styenc_hidden_dim', type=int, default=128)
    parser.add_argument('--styenc_lr', type=float, default=1e-4)
    parser.add_argument('--styenc_lr_pre', type=float, default=1e-4)
    parser.add_argument('--kl_weight', type=float, default=1e-3)
    parser.add_argument('--kl_weight_pre', type=float, default=1e-3)
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--eval_freq', type=int, default=1000)
    args = parser.parse_args()

    if not args.encode_style:
        args.styenc_embedding_dim = None
        args.styenc_code_dim = None
        args.styenc_hidden_dim = None

    with open('data/preprocessed.pkl', 'rb') as f:
        data = pickle.load(f)

    data_train, data_val, _ = utils.split_data(data)
    data_train = utils.arXivDataLoader(data_train, args.batch_size)
    data_val = utils.arXivDataLoader(data_val, args.batch_size)
    print("data loaded", flush=True)

    device = 'cuda'
    summarizer = Summarizer('google/pegasus-xsum').to(device)
    # summarizer = None
    style_encoder = None
    style_decoder = None
    if args.encode_style:
        style_encoder = StyleEncoder(
            embedding_dim=args.styenc_embedding_dim,
            code_dim=args.styenc_code_dim,
            num_layers=args.styenc_num_layers,
            hidden_dim=args.styenc_hidden_dim,
        ).to(device)
    if args.encode_style and args.pretrain:
        style_decoder = StyleDecoder(
            embedding=style_encoder.embedding,
            hidden_dim=args.styenc_hidden_dim,
            code_dim=args.styenc_code_dim,
        ).to(device)

    model = arXivModel(
        summarizer=summarizer,
        style_encoder=style_encoder,
        style_decoder=style_decoder,
    )
    print("model loaded", flush=True)

    mlflow.set_experiment(args.exp_name)
    run = mlflow.start_run()
    mlflow.log_params({
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'num_epochs_pre': args.num_epochs_pre,
        'num_beams': args.num_beams,
        'model_lr': args.model_lr,
        'encode_style': args.encode_style,
        'pretrain': args.pretrain,
        'styenc_embedding_dim': args.styenc_embedding_dim,
        'styenc_code_dim': args.styenc_code_dim,
        'styenc_num_layers': args.styenc_num_layers,
        'styenc_hidden_dim': args.styenc_hidden_dim,
        'styenc_lr': args.styenc_lr,
        'styenc_lr_pre': args.styenc_lr_pre,
        'kl_weight': args.kl_weight,
        'kl_weight_pre': args.kl_weight_pre,
    })

    if args.pretrain:
        print("pretraining start", flush=True)
        pretrain()

    # # lazy load
    # summarizer = Summarizer('google/pegasus-xsum').to(device)
    # model.set_summarizer(summarizer)
    print("training start", flush=True)
    train()

    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / 'checkpoint'
        model.save(path)
        mlflow.log_artifact(path)
