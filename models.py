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
        styenc_path=None,
        encode_style=False,
        styenc_embedding_dim=None,
        styenc_code_dim=None,
        styenc_hidden_dim=None,
    ):
        super().__init__()
        self.tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')

        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            model_name = checkpoint_path / 'model'

            with open(checkpoint_path / 'config.json', 'r') as f:
                config = json.load(f)

            self.encode_style = config['encode_style']
            self.styenc_embedding_dim = config['styenc_embedding_dim']
            self.styenc_code_dim = config['styenc_code_dim']
            self.styenc_hidden_dim = config['styenc_hidden_dim']
        else:
            self.encode_style = encode_style
            self.styenc_embedding_dim = styenc_embedding_dim
            self.styenc_code_dim = styenc_code_dim
            self.styenc_hidden_dim = styenc_hidden_dim

        self.model = PegasusForConditionalGeneration.from_pretrained(model_name)

        if self.encode_style:
            self.style_encoder = StyleEncoder(
                self.styenc_embedding_dim,
                self.styenc_code_dim,
                self.styenc_hidden_dim,
            )
            self.projector = nn.Linear(self.styenc_code_dim, self.model.config.d_model)

            if checkpoint_path is not None:
                styenc_path = checkpoint_path / 'style_encoder.pt'
                projector_path = checkpoint_path / 'projector.pt'
                self.style_encoder.load_state_dict(torch.load(styenc_path))
                self.projector.load_state_dict(torch.load(projector_path))
            elif styenc_path is not None:
                self.style_encoder.load_state_dict(torch.load(styenc_path))

    def encode(self, batch_raw, batch):
        batch_size = batch['input_ids'].shape[0]

        encoder_outputs = self.model.model.encoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
        )

        if self.encode_style:
            if 'title_pos' in batch_raw:
                # styenc_dist = self.get_styenc_dist(batch_raw['title_pos'])
                styenc_dist = self.style_encoder(batch_raw['title_pos'])
                styenc_sampled = styenc_dist.rsample()
            else:
                styenc_dist = None
                styenc_sampled = torch.randn(batch_size, self.styenc_dim, device=self.device)

            encoder_outputs.last_hidden_state = torch.cat([
                self.projector(styenc_sampled).unsqueeze(dim=1),
                encoder_outputs.last_hidden_state,
            ], dim=1)
            batch['attention_mask'] = torch.cat([
                torch.ones(batch_size, 1, dtype=batch['input_ids'].dtype, device=self.device),
                batch['attention_mask'],
            ], dim=1)
        else:
            styenc_dist = None
            styenc_sampled = None

        return encoder_outputs, styenc_dist, styenc_sampled

    def forward(self, batch_raw):
        batch = self.tokenizer.prepare_seq2seq_batch(
            src_texts=batch_raw['abstract'],
            tgt_texts=batch_raw['title'],
            return_tensors="pt",
        ).to(self.device)

        encoder_outputs, styenc_dist, styenc_sampled = self.encode(batch_raw, batch)

        outputs = self.model(
            encoder_outputs=encoder_outputs,
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )

        return outputs, styenc_dist, styenc_sampled

    @torch.no_grad()
    def generate(self, batch_raw, num_beams, decoder_start_token_id=None):
        batch = self.tokenizer.prepare_seq2seq_batch(
            src_texts=batch_raw['abstract'],
            return_tensors="pt",
        ).to(self.device)

        encoder_outputs, styenc_dist, styenc_sampled = self.encode(batch_raw, batch)

        max_length = self.model.config.max_length
        min_length = self.model.config.min_length
        pad_token_id = self.model.config.pad_token_id
        eos_token_id = self.model.config.eos_token_id
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
                'styenc_embedding_dim': self.styenc_embedding_dim,
                'styenc_code_dim': self.styenc_code_dim,
                'styenc_hidden_dim': self.styenc_hidden_dim,
            }, f, indent=4)

        self.model.save_pretrained(path / 'model')
        if self.encode_style:
            torch.save(self.style_encoder.state_dict(), path / 'style_encoder.pt')
            torch.save(self.projector.state_dict(), path / 'projector.pt')

    @property
    def device(self):
        return next(self.parameters()).device


class StyleEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim,
        code_dim,
        hidden_dim,
    ):
        super().__init__()

        with open('data/pos_list.json', 'r') as f:
            pos_list = json.load(f)

        self.pos_voc = {'<pad>': 0}
        self.pos_voc.update({pos: i + 1 for i, pos in enumerate(pos_list)})

        self.embedding = nn.Embedding(len(self.pos_voc), embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, code_dim * 2)

    def forward(self, title_pos):
        max_len = max([len(tp) for tp in title_pos])

        # pad title POS sequence
        title_pos_ids = []
        for tp in title_pos:
            pad_len = max_len - len(tp)
            tp_id = [self.pos_voc[p] for p in tp] + [self.pos_voc['<pad>']] * pad_len
            title_pos_ids.append(tp_id)

        title_pos_ids = torch.tensor(title_pos_ids, device=self.device).long()
        title_pos_embeds = self.embedding(title_pos_ids)

        _, hc = self.bilstm(title_pos_embeds)
        h = torch.cat([hc[0][0], hc[0][1]], dim=1)
        output = self.fc(h)

        mean, std = torch.chunk(output, 2, dim=1)
        std = F.softplus(std)
        dist = Normal(mean, std)
        return dist

    @property
    def device(self):
        return next(self.parameters()).device


class StyleDecoder(nn.Module):
    def __init__(
        self,
        embedding,
        code_dim,
        hidden_dim,
    ):
        super().__init__()

        with open('data/pos_list.json', 'r') as f:
            pos_list = json.load(f)

        self.pos_voc = {'<pad>': 0}
        self.pos_voc.update({pos: i + 1 for i, pos in enumerate(pos_list)})

        self.embedding = embedding
        self.lstm = nn.LSTM(self.embedding.embedding_dim, hidden_dim, batch_first=True)
        self.fc_in = nn.Linear(code_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, len(self.pos_voc))

    def forward(self, title_pos, code):
        max_len = max([len(tp) for tp in title_pos])

        # pad title POS sequence
        title_pos_ids = []
        masks = []
        for tp in title_pos:
            pad_len = max_len - len(tp)
            tp_id = [self.pos_voc['<pad>']] + \
                     [self.pos_voc[p] for p in tp] + \
                     [self.pos_voc['<pad>']] * pad_len
            title_pos_ids.append(tp_id)
            mask = [1.] * (len(tp) + 1) + [0.] * pad_len
            masks.append(mask)

        title_pos_ids = torch.tensor(title_pos_ids, device=self.device).long()
        title_pos_embeds = self.embedding(title_pos_ids)
        masks = torch.tensor(masks, device=self.device).float()

        h0 = self.fc_in(code).unsqueeze(dim=0)
        c0 = torch.zeros_like(h0)

        output, _ = self.lstm(title_pos_embeds, (h0, c0))
        logits = self.fc_out(output)

        title_pos_target = torch.cat([title_pos_ids[:, 1:], title_pos_ids[:, :1]], dim=1)
        loss = F.cross_entropy(
            logits.view(-1, len(self.pos_voc)),
            title_pos_target.view(-1),
            reduction='none'
        ).view(*masks.shape)
        loss = (loss * masks).mean()

        return loss

    @property
    def device(self):
        return next(self.parameters()).device
