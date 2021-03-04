from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch import optim


from word2vec.vocab import Word2VecDatamodule, get_data

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Multinomial

bce = F.binary_cross_entropy_with_logits


def one_hot(inds, vocab_size):
    return F.one_hot(inds, vocab_size).to(torch.float32)


def loss_func(pos, neg):
    dev = pos.device
    return (
        bce(p := pos.squeeze(), torch.ones(p.shape).to(dev)).sum()
        + bce(n := neg.squeeze(), torch.zeros(n.shape).to(dev)).sum()
    )


def loss_func_literal(pos: torch.Tensor, neg):
    p = pos.squeeze()
    n = neg.squeeze()
    return -torch.log(F.sigmoid(p)).mean() - torch.log(F.sigmoid(-n)).mean()


class NegativeSampler:
    def __init__(self, vocab_size, probabilities):
        assert len(probabilities) == vocab_size
        self.dist = Multinomial(1, probs=probabilities)

    def __call__(self, shape):
        return torch.where(self.dist.sample(shape))[-1]


class SkipGramNegSampling(LightningModule):
    def __init__(self, vocab_size, embedding_dim, lr):
        super().__init__()
        self.save_hyperparameters()
        self.U = nn.Linear(self.hparams.vocab_size, self.hparams.embedding_dim)
        self.V = nn.Linear(self.hparams.vocab_size, self.hparams.embedding_dim)

    def forward(self, center, ctx, neg):
        U = self.U
        V = self.V

        v = V(one_hot(center, V.in_features))[:, None, :].transpose(1, 2)
        u = U(one_hot(ctx, U.in_features))
        ũ = U(one_hot(neg, U.in_features))

        return (u @ v, ũ @ v)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

    def _step(self, stage, batch):
        pos, neg = self(*batch)
        loss = loss_func(pos, neg)
        self.log(f"{stage}_loss", loss)
        return loss

    def training_step(self, batch, _):
        return self._step("train", batch)

    def validation_step(self, batch, _):
        return self._step("val", batch)

    def test_step(self, batch, _):
        return self._step("test", batch)


def negative_sampling_main(
    lr=1e-3,
    context_size=2,
    neg_per_pos=5,
    embedding_dim=100,
    n_epochs=1,
    batch_size=32,
    num_workers=0,
    train_path="data/wikitext-2/wiki.train.tokens",
    val_path="data/wikitext-2/wiki.valid.tokens",
    test_path="data/wikitext-2/wiki.test.tokens",
):
    docs, vocab = get_data(train_path, val_path, test_path)
    assert vocab is not None
    negative_sampler = NegativeSampler(
        len(vocab),
        torch.tensor([vocab.freqs[tok] for tok in vocab.itos], dtype=torch.float32),
    )

    lm = SkipGramNegSampling(len(vocab), embedding_dim, lr=lr)
    dm = Word2VecDatamodule(
        docs,
        vocab,
        context_size,
        negative_sampler,
        k=neg_per_pos,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if torch.cuda.is_available():
        gpus = -1
    else:
        gpus = 0

    ckpt = ModelCheckpoint(monitor="val_loss")
    trainer = Trainer(
        max_epochs=n_epochs,
        gpus=gpus,
        log_every_n_steps=200,
        flush_logs_every_n_steps=400,
        checkpoint_callback=ckpt,
        distributed_backend="ddp",
    )
    trainer.fit(lm, datamodule=dm)


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--context-size", type=int, default=2)
    parser.add_argument("--neg-per-pos", type=int, default=5)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--train-path", default="data/wikitext-2/wiki.train.tokens")
    parser.add_argument("--val-path", default="data/wikitext-2/wiki.valid.tokens")
    parser.add_argument("--test-path", default="data/wikitext-2/wiki.test.tokens")
    args = parser.parse_args()

    negative_sampling_main(
        lr=args.lr,
        context_size=args.context_size,
        neg_per_pos=args.neg_per_pos,
        embedding_dim=args.embedding_dim,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
    )
