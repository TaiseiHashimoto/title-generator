import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from transformers import (
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    BartConfig,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    BeamSearchScorer,
)
from transformers.models.bart.modeling_bart import BartEncoder
import json
from pathlib import Path


class arXivModel(nn.Module):
    def __init__(
        self,
        model_name=None,
        checkpoint_path=None,
        encode_style=False,
        style_encoder_layer=None,
        style_encoder_dim=None,
        style_encoder_ffn_dim=None,
        style_encoder_head=None,
        device='cuda',
    ):
        super().__init__()
        self.tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')

        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            model_name = checkpoint_path / 'model'

            with open(checkpoint_path / 'config.json', 'r') as f:
                config = json.load(f)
            self.encode_style = config['encode_style']
            self.style_encoder_layer = config['style_encoder_layer']
            self.style_encoder_dim = config['style_encoder_dim']
            self.style_encoder_ffn_dim = config['style_encoder_ffn_dim']
            self.style_encoder_head = config['style_encoder_head']
            self.device = config['device']
        else:
            self.encode_style = encode_style
            self.style_encoder_layer = style_encoder_layer
            self.style_encoder_dim = style_encoder_dim
            self.style_encoder_ffn_dim = style_encoder_ffn_dim
            self.style_encoder_head = style_encoder_head
            self.device = device

        self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

        if self.encode_style:
            with open('data/pos_list.json', 'r') as f:
                pos_list = json.load(f)

            self.pos_voc = {'<mean>': 0, '<std>': 1, '<pad>': 2}
            self.pos_voc.update({pos: i + 3 for i, pos in enumerate(pos_list)})

            if checkpoint_path is not None:
                style_encoder_path = checkpoint_path / 'style_encoder'
                self.style_encoder = BartEncoder.from_pretrained(style_encoder_path).to(device)
            else:
                config = BartConfig(
                    encoder_layers=self.style_encoder_layer,
                    d_model=self.style_encoder_dim,
                    encoder_ffn_dim=self.style_encoder_ffn_dim,
                    encoder_attention_heads=self.style_encoder_head,
                    vocab_size=len(self.pos_voc),
                    max_position_embeddings=50,
                    pad_token_id=self.pos_voc['<pad>'],
                )
                self.style_encoder = BartEncoder(config).to(device)

            self.projector = nn.Linear(self.style_encoder_dim, self.model.config.d_model).to(device)
            if checkpoint_path is not None:
                projector_path = checkpoint_path / 'projector.pt'
                self.projector.load_state_dict(torch.load(projector_path))

    def get_style_encoder_dist(self, title_pos):
        max_len = max([len(tp) for tp in title_pos])

        # pad title POS sequence
        title_pos_ids = []
        attention_mask = []
        for tp in title_pos:
            pad_len = max_len - len(tp)
            tp_id = [self.pos_voc['<mean>'], self.pos_voc['<std>']] + \
                      [self.pos_voc[p] for p in tp] + \
                      [self.pos_voc['<pad>']] * (pad_len + 1)
            title_pos_ids.append(tp_id)

            atm = [1] * (len(tp) + 3) + [0] * pad_len
            attention_mask.append(atm)

        title_pos_ids = torch.tensor(title_pos_ids, device=self.device).long()
        attention_mask = torch.tensor(attention_mask, device=self.device).long()
        output = self.style_encoder(input_ids=title_pos_ids, attention_mask=attention_mask)

        mean = output.last_hidden_state[:, 0]
        std = F.softplus(output.last_hidden_state[:, 1])
        dist = Normal(mean, std)
        return dist

    def encode(self, batch_raw, batch):
        batch_size = batch['input_ids'].shape[0]

        encoder_outputs = self.model.model.encoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
        )

        if self.encode_style:
            if 'title_pos' in batch_raw:
                style_encoder_dist = self.get_style_encoder_dist(batch_raw['title_pos'])
                style_encoder_sampled = style_encoder_dist.rsample()
            else:
                style_encoder_sampled = torch.randn(batch_size, self.style_encoder_dim)

            encoder_outputs.last_hidden_state = torch.cat([
                self.projector(style_encoder_sampled).unsqueeze(dim=1),
                encoder_outputs.last_hidden_state,
            ], dim=1)
            batch['attention_mask'] = torch.cat([
                torch.ones(batch_size, 1, dtype=batch['input_ids'].dtype, device=self.device),
                batch['attention_mask'],
            ], dim=1)
        else:
            style_encoder_dist = None
            style_encoder_sampled = None

        return encoder_outputs, style_encoder_dist, style_encoder_sampled

    def forward(self, batch_raw):
        batch = self.tokenizer.prepare_seq2seq_batch(
            src_texts=batch_raw['abstract'],
            tgt_texts=batch_raw['title'],
            return_tensors="pt",
        ).to(self.device)

        encoder_outputs, style_encoder_dist, style_encoder_sampled = self.encode(batch_raw, batch)

        outputs = self.model(
            encoder_outputs=encoder_outputs,
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )

        return outputs, style_encoder_dist, style_encoder_sampled

    @torch.no_grad()
    def generate(self, batch_raw, num_beams, decoder_start_token_id=None):
        batch = self.tokenizer.prepare_seq2seq_batch(
            src_texts=batch_raw['abstract'],
            return_tensors="pt",
        ).to(self.device)

        encoder_outputs, style_encoder_dist, style_encoder_sampled = self.encode(batch_raw, batch)

        max_length = self.model.config.max_length
        min_length = self.model.config.min_length
        pad_token_id = self.model.config.pad_token_id
        eos_token_id = self.model.config.eos_token_id
        bos_token_id = self.model.config.bos_token_id
        batch_size = batch['input_ids'].shape[0]

        logits_processor = LogitsProcessorList()
        logits_processor.append(MinLengthLogitsProcessor(min_length, eos_token_id))

        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=num_beams,
            device=self.device,
        )

        if decoder_start_token_id is None:
            # BartModel is trained with initial_token='<eos>' (bug?)
            # Therefore, I use '<eos>' also for generation (default: '<bos>')
            decoder_start_token_id = eos_token_id

        # expand inputs for beam search
        input_ids = torch.ones((num_beams * batch_size, 1),
                                    dtype=batch['input_ids'].dtype,
                                    device=self.device) * decoder_start_token_id
        attention_mask = batch['attention_mask'].repeat_interleave(num_beams, dim=0)
        encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.repeat_interleave(num_beams, dim=0)

        predicted = self.model.beam_search(
            input_ids,
            beam_scorer,
            logits_processor=logits_processor,
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
        )
        generated = self.tokenizer.batch_decode(predicted, skip_special_tokens=True)
        return generated

    def save(self, path):
        path = Path(path)
        path.mkdir(exist_ok=True)

        with open(path / 'config.json', 'w') as f:
            json.dump({
                'encode_style': self.encode_style,
                'style_encoder_layer': self.style_encoder_layer,
                'style_encoder_dim': self.style_encoder_dim,
                'style_encoder_ffn_dim': self.style_encoder_ffn_dim,
                'style_encoder_head': self.style_encoder_head,
                'device': self.device,
            }, f, indent=4)

        self.model.save_pretrained(path / 'model')
        if self.encode_style:
            self.style_encoder.save_pretrained(path / 'style_encoder')
            torch.save(self.projector.state_dict(), path / 'projector.pt')
