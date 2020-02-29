# coding=utf8

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchcrf import CRF
from pytorch_pretrained_bert import BertModel


class Net(nn.Module):
    def __init__(self, top_rnns=False, vocab_size=None, device='cpu',
                 finetuning=False, dropout=0., lin_dim=128):
        super(Net, self).__init__()
        self.bert = BertModel.from_pretrained(os.getenv('BERT_BASE_CHINESE', 'bert-base-chinese'))
        self.top_rnns = top_rnns
        if self.top_rnns:
            self.rnn = nn.LSTM(bidirectional=True, num_layers=2,
                               input_size=768, hidden_size=lin_dim // 2,
                               batch_first=True, dropout=dropout)
        self.fc = nn.Linear(lin_dim, vocab_size)
        self.vocab_size = vocab_size
        self.device = device
        self.finetuning = finetuning

    def forward(self, x, y=None):
        x = x.to(self.device)
        mask = x.eq(0).unsqueeze(-1)
        if self.finetuning and self.trianing:
            self.bert.train()
            encoded_layers, _ = self.bert(x)
            enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(x)
                enc = encoded_layers[-1]

        if self.top_rnns:
            enc, _ = self.rnn(enc)
        if mask is not None:
            mask = mask.to(self.device)
            enc = enc.masked_fill(mask, 0)

        logits = self.fc(enc)
        y_hat = logits.argmax(-1)
        if y is None:
            return logits, y_hat
        else:
            y = y.to(self.device)
            loss = self._calc_loss(logits, y)
            return logits, y_hat, loss

    def _calc_loss(self, logits, y):
        logits = logits.view(-1, logits.shape[-1])
        y = y.view(-1)
        loss = F.cross_entropy(logits, y, ignore_index=0)
        return loss


class NetCRF(Net):
    def __init__(self, *args, **kwargs):
        super(NetCRF, self).__init__(*args, **kwargs)
        self.crf = CRF(self.vocab_size, batch_first=True)

    def forward(self, x, y=None):
        feats, _ = super(NetCRF, self).forward(x, y)[0]
        crf_mask = x.ne(0)
        y_hat = self.crf.decode(feats, mask=crf_mask)
        if y is None:
            return feats, y_hat
        else:
            y = y.to(self.device)
            loss = self._calc_loss(feats, y)
            return feats, y_hat, loss

    def _calc_loss(self, logits, y):
        loss = -self.crf(logits, y, mask=y.ne(0), reduction="token_mean")
        return loss
