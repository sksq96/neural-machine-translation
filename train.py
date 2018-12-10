import time
import os
from argparse import ArgumentParser


import torch
from torchtext import data, datasets

from constants import *
from model import make_model, LabelSmoothing, NoamOpt, Batch, SimpleLossCompute, MyIterator
from util import rebatch


# training settings
parser = ArgumentParser(description='Transformer for Language Translation')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--encoder', type=str, default='rnn')
parser.add_argument('--n_layers', type=int, default=6)
parser.add_argument('--max_len', type=int, default=100)
parser.add_argument('--min_freq', type=int, default=3)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--basepath', type=str, default='iwslt-vi-en')
parser.add_argument('--train_file', type=str, default='train.tok.csv')
parser.add_argument('--val_file', type=str, default='dev.tok.csv')
parser.add_argument('--log_every', type=int, default=50)
parser.add_argument('--experiment', type=str, default='test')
parser.add_argument('--outf', type=str, default='save')
params = parser.parse_args()

print(params)


try:
    os.makedirs(params.outf)
except OSError:
    pass

# gpu business
if torch.cuda.is_available():
    torch.cuda.set_device(params.gpu)
    device = torch.device('cuda:{}'.format(params.gpu))
else:
    device = torch.device('cpu')


# define text felids
SRC = data.Field(pad_token=BLANK_WORD)
TGT = data.Field(init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=BLANK_WORD)


# train_file = filename.format('train', 'csv')
# val_file = filename.format('dev', 'csv')

train, val = data.TabularDataset.splits(
    path=params.basepath,
    train=params.train_file, validation=params.val_file,
    format='tsv',
    skip_header=True,
    fields=[('src', SRC), ('tgt', TGT)],
    filter_pred=lambda x: len(vars(x)['src']) <= params.max_len and len(
        vars(x)['tgt']) <= params.max_len,
)


# build vocabulary
SRC.build_vocab(train.src, min_freq=params.min_freq)
TGT.build_vocab(train.tgt, min_freq=params.min_freq)

pad_idx = TGT.vocab.stoi[BLANK_WORD]

len_source =  len(SRC.vocab)
len_target = len(TGT.vocab)
print(f"Number of words in source vocab: {len_source}")
print(f"Number of words in target vocab: {len_target}")


# train_iter, valid_iter = data.BucketIterator.splits(
#     (train, val),
#     batch_size=params.batch_size,
#     sort_key=lambda x: (len(x.src), len(x.tgt)),
#     device=device
# )

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.tgt) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


train_iter = MyIterator(train, batch_size=params.batch_size, device=device,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.tgt)),
                        batch_size_fn=batch_size_fn, train=True)
valid_iter = MyIterator(val, batch_size=params.batch_size, device=device,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.tgt)),
                        batch_size_fn=batch_size_fn, train=False)


model = make_model(len_source, len_target, N=params.n_layers)
model.to(device)


criterion = LabelSmoothing(
    size=len_target, padding_idx=pad_idx, smoothing=0.1)
criterion.to(device)

model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def run_epoch(epoch, data_iter, model, loss_compute, train=True):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.
    for i, batch in enumerate(data_iter):
        ntokens = batch.ntokens.float()
        out = model.forward(batch.src, batch.tgt,
                            batch.src_mask, batch.tgt_mask)
        loss = loss_compute(out, batch.tgt_y, ntokens)
        total_loss += loss
        total_tokens += ntokens
        tokens += ntokens
        if i % params.log_every == 0 and train:
            elapsed = time.time() - start
            print(f"Epoch {epoch} Step: {i} Loss: {loss / ntokens.float()} Tokens per Sec")
            start = time.time()
            tokens = 0.
    return total_loss / total_tokens


def main():
    for epoch in range(params.epochs):
        model.train()
        run_epoch(epoch, (rebatch(pad_idx, b) for b in train_iter), model,
                  SimpleLossCompute(model.generator, criterion, opt=model_opt))
        model.eval()
        loss = run_epoch(epoch, (rebatch(pad_idx, b) for b in valid_iter), model,
                         SimpleLossCompute(model.generator, criterion, opt=None))
        print(loss.data())

        # do checkpointing
        torch.save(model.state_dict(), f'{params.outf}/model.pth')


if __name__ == "__main__":
    main()

