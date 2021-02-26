import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from word2vec.data import Word2VecDatamodule, get_data
from word2vec.negative_sampling import NegativeSampler, SkipGramNegSampling


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

    num_workers = int(sys.argv[1])
    negative_sampling_main(num_workers=num_workers, batch_size=128, n_epochs=10)
