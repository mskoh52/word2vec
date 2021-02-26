import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn, optim
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
        self.U = nn.Linear(vocab_size, embedding_dim)
        self.V = nn.Linear(vocab_size, embedding_dim)
        self.lr = lr

    def forward(self, center, ctx, neg):
        U = self.U
        V = self.V

        v = V(one_hot(center, V.in_features))[:, None, :].transpose(1, 2)
        u = U(one_hot(ctx, U.in_features))
        ũ = U(one_hot(neg, U.in_features))

        return (u @ v, ũ @ v)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

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
