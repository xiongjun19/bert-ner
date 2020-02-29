# coding=utf8

import random
import torch
import numpy as np
from torch.utils.data.dataset import Dataset


MAX_SEQ_LENGTH = 512


class DataMixin(Dataset):
    def __init__(self):
        super(DataMixin, self).__init__()

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        if isinstance(index, slice):
            begin, end, step = index.indices(len(self))
            return [self.get_example(i) for i in range(begin, end, step)]
        elif isinstance(index, list):
            return [self.get_example(i) for i in index]
        else:
            return self.get_example(index)

    def __len__(self):
        raise NotImplementedError

    def get_example(self, i):
        raise NotImplementedError


class NerDataset(DataMixin):
    VOCAB = ['<PAD>', "O", "B-DATE", "I-DATE", "B-METRIC", "I-METRIC", "B-NUMBER", "I-NUMBER", "B-ENTITY", "I-ENTITY",
             "[CLS]", "[SEP]", "X"]
    START_TAG = '[CLS]'
    STOP_TAG = '[SEP]'
    tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
    idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}

    def __init__(self, tokenizer, texts, labels):
        super(NerDataset, self).__init__()
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def get_example(self, i):
        sen_words = self.texts[i]
        tags = self.labels[i]
        sen_words = [self.START_TAG] + sen_words + [self.STOP_TAG]
        tags = [self.START_TAG] + tags + [self.STOP_TAG]

        # we give credits only to the first piece.
        x, y = [], []  # list of ids
        is_heads = []

        for word, tag in zip(sen_words, tags):
            tokens = self.tokenizer.tokenize(word) \
                if word not in (self.START_TAG, self.STOP_TAG) else [word]
            if not tokens:
                tokens = ["[UNK]"]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            is_head = [1] + [0] * (len(tokens) - 1)
            sub_tags = [tag] + ['X'] * (len(tokens) - 1)
            sub_tag_ids = [self.tag2idx[x] for x in sub_tags]
            x.extend(token_ids)
            y.extend(sub_tag_ids)
            is_heads.extend(is_head)
        assert len(x) == len(y) == len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"
        seq_len = len(y)
        words = " ".join(sen_words)
        tags = " ".join(tags)
        return words, x, is_heads, tags, y, seq_len


def prepare_data(f_path, sep="\t"):
    texts = []
    labels = []
    with open(f_path) as in_:
        block = in_.read()
        line_arr = block.split("\n\n")
        random.seed(777)
        random.shuffle(line_arr)
        for line in line_arr:
            raw_info = _decode_raw(line, sep)
            if raw_info is not None:
                tmp_char_list, tmp_tag_list = raw_info
                if len(tmp_char_list) > 0:
                    texts.append(tmp_char_list)
                    labels.append(tmp_tag_list)
    return texts, labels


def _decode_raw(line, sep):
    line = line.strip()
    if len(line) <= 0:
        return None
    line_arr = line.split("\n")
    char_list = []
    tag_list = []
    for sub_line in line_arr:
        sub_line = sub_line.strip()
        if len(sub_line) > 0:
            sub_arr = sub_line.split(sep)
            char, tag = sub_arr[0], sub_arr[1]
            char_list.append(char)
            tag_list.append(tag)
    return char_list, tag_list


def pad(batch):
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()
    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # 0: <pad>
    x = f(1, maxlen)
    y = f(-2, maxlen)
    f = torch.LongTensor
    return words, f(x), is_heads, tags, f(y), seqlens


def trunk_batch(batch):
    words, x, is_heads, tags, y, seqlens = batch

    if x.size(1) > MAX_SEQ_LENGTH:
        x = x[:, : MAX_SEQ_LENGTH]
        is_heads = list(map(lambda x: x[: MAX_SEQ_LENGTH], is_heads))

        _tags = []
        for t in tags:
            _t = t.split()
            if len(_t) > MAX_SEQ_LENGTH:
                _t = _t[: MAX_SEQ_LENGTH]
            _tags.append(' '.join(_t))
        tags = _tags

        _words = []
        for w in words:
            _w = w.split()
            if len(_w) > MAX_SEQ_LENGTH:
                _w = _w[: MAX_SEQ_LENGTH]
            _words.append(' '.join(_w))
        words = _words

        y = y[:, : MAX_SEQ_LENGTH]

        for i, elem in enumerate(seqlens):
            if elem > MAX_SEQ_LENGTH:
                seqlens[i] = MAX_SEQ_LENGTH
    return words, x, is_heads, tags, y, seqlens