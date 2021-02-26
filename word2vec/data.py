import re
from collections import Counter

import torch
from pytorch_lightning import LightningDataModule
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchtext.vocab import Vocab

nlp = English()
tokenizer = Tokenizer(nlp.vocab)


class PadToken:
    def __init__(self, orth_):
        self.orth_ = orth_


class Word2VecDataset(Dataset):
    def __init__(self, tokens, context_size, pad_token="<pad>"):
        self.tokens = tokens
        self.context_size = context_size
        self.pad_token = PadToken(pad_token)

        self._ix = 0

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, ix):
        start = max(0, ix - self.context_size)
        end = min(len(self.tokens), ix + self.context_size + 1)
        pre_ctx = self.tokens[start:ix]
        post_ctx = self.tokens[ix + 1 : end]
        ctx = list(pre_ctx) + list(post_ctx)
        pads_needed = 2 * self.context_size - len(ctx)
        for _ in range(pads_needed):
            ctx.append(self.pad_token)

        return self.tokens[ix], ctx


class Word2VecDatamodule(LightningDataModule):
    def __init__(
        self,
        docs,
        vocab,
        context_size,
        negative_sampler,
        k: int,
        batch_size,
        num_workers=0,
    ):
        super().__init__()
        self.docs = docs
        self.vocab = vocab
        self.context_size = context_size
        self.negative_sampler = negative_sampler
        self.k = k
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _collate(self, batch):
        return (
            torch.tensor([self.vocab.stoi[t[0].orth_] for t in batch]),
            torch.tensor([[self.vocab.stoi[t.orth_] for t in x[1]] for x in batch]),
            self.negative_sampler((len(batch), self.context_size, self.k)),
        )

    def train_dataloader(self):
        return DataLoader(
            Word2VecDataset(self.docs["train"], self.context_size),
            collate_fn=self._collate,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            Word2VecDataset(self.docs["val"], self.context_size),
            collate_fn=self._collate,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            Word2VecDataset(self.docs["test"], self.context_size),
            collate_fn=self._collate,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


def get_data(
    train_path="data/wikitext-2/wiki.train.tokens",
    val_path="data/wikitext-2/wiki.valid.tokens",
    test_path="data/wikitext-2/wiki.test.tokens",
):
    paths = {"train": train_path, "val": val_path, "test": test_path}
    docs = {}
    vocab = None
    for stage, path in paths.items():
        txt = open(path).read().strip()
        txt = re.sub(r"\n\s*\n\s*", "\n", txt)
        doc = tokenizer(txt)
        docs[stage] = doc
        if stage == "train":
            vocab = Vocab(Counter(map(str, doc)))

    return docs, vocab
