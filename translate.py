import os
from argparse import ArgumentParser

import torch
from constants import *
from torchtext import data

from model import greedy_decode, make_model
from util import rebatch

parser = ArgumentParser(description='Transformer for Language Translation')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_len', type=int, default=100)
parser.add_argument('--min_freq', type=int, default=3)
parser.add_argument('--n_layers', type=int, default=6)
parser.add_argument('--basepath', type=str, default='iwslt-vi-en')
parser.add_argument('--train_file', type=str, default='train.tok.csv')
parser.add_argument('--val_file', type=str, default='dev.tok.csv')
parser.add_argument('--test_file', type=str, default='test.tok.csv')
parser.add_argument('--savepath', type=str, default='save')
parser.add_argument('-model_file', type=str, default='netG_epoch_49.pth')
params = parser.parse_args()

# print(params)


# define text felids
SRC = data.Field(pad_token=BLANK_WORD)
TGT = data.Field(init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=BLANK_WORD)


train, val, test = data.TabularDataset.splits(
    path=params.basepath,
    train=params.train_file, validation=params.val_file, test=params.test_file,
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

len_source = len(SRC.vocab)
len_target = len(TGT.vocab)
# print(f"Number of words in source vocab: {len_source}")
# print(f"Number of words in target vocab: {len_target}")


test_iter = data.BucketIterator(test, batch_size=params.batch_size, shuffle=False)
# test_iter = (rebatch(pad_idx, b) for b in test_iter)


model = make_model(len_source, len_target, N=params.n_layers)
model.load_state_dict(
    torch.load(os.path.join(params.savepath, params.model_file), map_location='cpu')
)


def main(data_iter):
    # f = open(f"generated.{params.test_file}", 'w')
    for i, batch in enumerate(data_iter):
        src = batch.src.transpose(0, 1)[:1]
        src_mask = (src != SRC.vocab.stoi[BLANK_WORD]).unsqueeze(-2)
        out = greedy_decode(model, src, src_mask, max_len=params.max_len, start_symbol=TGT.vocab.stoi[BOS_WORD])
        for i in range(1, out.size(1)):
            sym = TGT.vocab.itos[out[0, i]]
            if sym == EOS_WORD:
                break
            print(sym, end=" ")
        print()



if __name__ == "__main__":
    main(test_iter)
