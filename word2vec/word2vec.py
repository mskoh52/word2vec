import torch.nn.functional as F
from torch import nn


class SkipGram(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim

        self.U = nn.Linear(vocab_size, dim)
        self.V = nn.Linear(vocab_size, dim)

    def forward(self, center, context):
        """compute score for center/context pairs

        center is N x 1
        context is N x K where K is context length (center +/- K)
        """
        v = self.V(F.one_hot(center, self.vocab_size))
        u = self.U(F.one_hot(context, self.vocab_size))
        z = u @ v
        return z


def softmax_skip_gram(z, ctx):
    return F.cross_entropy(z.repeat((len(ctx), 1)), ctx)
