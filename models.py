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


# save all parameters
PegasusForConditionalGeneration._keys_to_ignore_on_save = []


class arXivModel():
    def __init__(
        self,
        summarizer,
        style_encoder=None,
    ):
        self.summarizer = summarizer
        self.style_encoder = style_encoder

        self.summarizer._keys_to_ignore_on_load_missing = []
        self.summarizer._keys_to_ignore_on_save = []

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device='cuda'):
        summarizer = Summarizer.from_checkpoint(checkpoint_path).to(device)
        style_encoder = StyleEncoder.from_checkpoint(checkpoint_path)
        if style_encoder is not None:
            style_encoder.to(device)
        return cls(summarizer, style_encoder)

    def get_style_code(self, batch_raw):
        if self.style_encoder is not None:
            style_code_dist = self.style_encoder(batch_raw)
            style_code = style_code_dist.rsample()
            return style_code_dist, style_code
        else:
            return None, None

    def forward_train(self, batch_raw):
        style_code_dist, style_code = self.get_style_code(batch_raw)
        scales = self.style_encoder.scales if self.style_encoder is not None else None
        outputs = self.summarizer(batch_raw, style_code, scales)
        return outputs.loss, style_code_dist, style_code

    @torch.no_grad()
    def generate(self, batch_raw, style_code=None, num_beams=None):
        if style_code is None:
            style_code_dist, style_code = self.get_style_code(batch_raw)
        scales = self.style_encoder.scales if self.style_encoder is not None else None
        generated = self.summarizer.generate(batch_raw, style_code, scales, num_beams=num_beams)
        return generated

    def save(self, checkpoint_path):
        self.summarizer.save(checkpoint_path)
        if self.style_encoder is not None:
            self.style_encoder.save(checkpoint_path)


class Summarizer(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name)

    @classmethod
    def from_checkpoint(cls, checkpoint_path):
        return cls(Path(checkpoint_path, 'model'))

    def encode(self, batch, style_code=None, scales=None):
        encoder_outputs = self.model.model.encoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
        )

        if style_code is not None:
            style_code_rep = style_code.repeat(1, self.model.config.d_model // style_code.shape[1])
            encoder_outputs.last_hidden_state = torch.cat([
                style_code_rep.unsqueeze(dim=1) * scales[1],
                encoder_outputs.last_hidden_state,
            ], dim=1)
            batch['attention_mask'] = torch.cat([
                torch.ones_like(batch['attention_mask'][:, :1]),
                batch['attention_mask'],
            ], dim=1)

        return encoder_outputs

    def forward(self, batch_raw, style_code=None, scales=None):
        batch = self.tokenizer.prepare_seq2seq_batch(
            src_texts=batch_raw['abstract'],
            tgt_texts=batch_raw['title'],
            max_length=self.tokenizer.model_max_length-1,  # save room for style code
            return_tensors="pt",
        ).to('cuda')

        encoder_outputs = self.encode(batch, style_code, scales)

        outputs = self.model(
            encoder_outputs=encoder_outputs,
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        return outputs

    @torch.no_grad()
    def generate(self, batch_raw, style_code=None, scales=None, num_beams=None, decoder_start_token_id=None):
        batch = self.tokenizer.prepare_seq2seq_batch(
            src_texts=batch_raw['abstract'],
            max_length=self.tokenizer.model_max_length-1,  # save room for style code
            return_tensors="pt",
        ).to('cuda')

        encoder_outputs = self.encode(batch, style_code, scales)

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
            device='cuda',
        )

        if decoder_start_token_id is None:
            # BartModel is trained with initial_token='<eos>' (bug?)
            # Therefore, I use '<eos>' also for generation (default: '<bos>')
            decoder_start_token_id = eos_token_id

        # expand inputs for beam search
        input_ids = torch.ones((num_beams * batch_size, 1),
                                    dtype=batch['input_ids'].dtype,
                                    device='cuda') * decoder_start_token_id
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

    def save(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.mkdir(exist_ok=True)
        self.model.save_pretrained(checkpoint_path / 'model')


class StyleEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim,
        code_dim,
        num_layers,
        hidden_dim,
    ):
        super().__init__()
        with open(Path('data', 'pos_list.json'), 'r') as f:
            pos_list = json.load(f)
            self.pos_voc = {'<pad>': 0}
            self.pos_voc.update({pos: i + 1 for i, pos in enumerate(pos_list)})

        self.embedding_dim = embedding_dim
        self.code_dim = code_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(len(self.pos_voc), embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear((hidden_dim * 2) * num_layers, code_dim * 2)
        # TODO: value ok?
        self.log_scales = nn.Parameter(torch.tensor([3., 0.15]).log())

    @classmethod
    def from_checkpoint(cls, checkpoint_path):
        path = Path(checkpoint_path, 'styenc')
        if not path.exists():
            return None

        with open(path / 'config.json', 'r') as f:
            config = json.load(f)
            embedding_dim = config['embedding_dim']
            code_dim = config['code_dim']
            hidden_dim = config['hidden_dim']
            # num_layers = config['num_layers']
            num_layers = 1

        encoder = cls(
            embedding_dim=embedding_dim,
            code_dim=code_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        encoder.load_state_dict(torch.load(path / 'model.pt'))
        return encoder

    @property
    def scales(self):
        scales = self.log_scales.exp()
        scales = torch.clamp(scales, min=0.05, max=10.)
        return scales

    def forward(self, batch_raw):
        title_pos = batch_raw['title_pos']
        max_len = max([len(tp) for tp in title_pos])

        # pad title POS sequence
        title_pos_ids = []
        for tp in title_pos:
            pad_len = max_len - len(tp)
            tp_id = [self.pos_voc[p] for p in tp] + [self.pos_voc['<pad>']] * pad_len
            title_pos_ids.append(tp_id)

        title_pos_ids = torch.tensor(title_pos_ids, device='cuda').long()
        title_pos_embeds = self.embedding(title_pos_ids)

        _, hc = self.lstm(title_pos_embeds)
        # (num_layers * num_directions, batch_size, hidden_dim)
        h = hc[0]
        hidden = h.permute(1, 0, 2).reshape(-1, self.num_layers * 2 * self.hidden_dim)
        output = self.fc(hidden)

        mean, std = torch.chunk(output, 2, dim=1)
        std = F.softplus(std)
        return Normal(mean, std)

    def save(self, checkpoint_path):
        path = Path(checkpoint_path, 'styenc')
        path.mkdir(parents=True, exist_ok=True)

        with open(path / 'config.json', 'w') as f:
            json.dump({
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'code_dim': self.code_dim,
            }, f, indent=4)

        torch.save(self.state_dict(), path / 'model.pt')
