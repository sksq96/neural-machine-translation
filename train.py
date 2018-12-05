import time
from argparse import ArgumentParser


import torch
from torchtext import data, datasets

from constants import *
from model import make_model, LabelSmoothing, NoamOpt, Batch, SimpleLossCompute, MyIterator
from util import rebatch


# training settings
parser = ArgumentParser(description='Transformer for Language Translation')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--encoder', type=str, default='rnn')
# TODO: link size of d_embed to the pre-trained word-embedding
parser.add_argument('--d_embed', type=int, default=300)
parser.add_argument('--d_hidden', type=int, default=300)
parser.add_argument('--d_fc', type=int, default=100)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--lr', type=float, default=.001)
parser.add_argument('--dp_ratio', type=float, default=0.2)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--basepath', type=str, default='iwslt-vi-en')
parser.add_argument('--train_file', type=str, default='train_tok.csv')
parser.add_argument('--val_file', type=str, default='val_tok.csv')
parser.add_argument('--log_every', type=int, default=50)
parser.add_argument('--dev_every', type=int, default=1000)
parser.add_argument('--experiment', type=str, default='test')
params = parser.parse_args()


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
    train=train_file, validation=val_file,
    format='tsv',
    skip_header=True,
    fields=[('src', SRC), ('tgt', TGT)])


# build vocabulary
MIN_FREQ = 1
SRC.build_vocab(train.src, min_freq=MIN_FREQ)
TGT.build_vocab(train.tgt, min_freq=MIN_FREQ)

pad_idx = TGT.vocab.stoi[BLANK_WORD]

len_source =  len(SRC.vocab)
len_target = len(TGT.vocab)
print(f"Number of words in source vocab: {len_source}")
print(f"Number of words in target vocab: {len_target}")


train_iter, valid_iter = data.BucketIterator.splits(
    (train, val),
    batch_size=params.batch_size,
    sort_key=lambda x: (len(x.src), len(x.tgt)),
    device=device)


model = make_model(len_source, len_target, N=params.n_layers)
model.to(device)


criterion = LabelSmoothing(
    size=len_target, padding_idx=pad_idx, smoothing=0.1)
criterion.to(device)

model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))



def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt,
                            batch.src_mask, batch.tgt_mask)
        loss = loss_compute(out, batch.tgt_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % params.log_every == 0:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.ntokens.float(), tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


def main():
    for epoch in range(params.epochs):
        model.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter), model,
                  SimpleLossCompute(model.generator, criterion, opt=model_opt))
        model.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), model,
                         SimpleLossCompute(model.generator, criterion, opt=None))
        print(loss)



if __name__ == "__main__":
    main()

