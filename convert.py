import os
import pandas as pd
from argparse import ArgumentParser


parser = ArgumentParser(description='Transformer for Language Translation')
parser.add_argument('--basepath', type=str, default='iwslt-vi-en')
# parser.add_argument('--train_file', type=str, default='train_tok.csv')
# parser.add_argument('--val_file', type=str, default='val_tok.csv')
params = parser.parse_args()


languages = params.basepath.split('-')[-2:]

names = ['src', 'tgt']
filename = '{}.tok.{}'

for t in ['dev', 'train', 'test']:
    paths = [os.path.join(params.basepath, filename.format(t, l)) for l in languages]

    # read data from multiple files in one df
    df = pd.concat(
        [pd.read_csv(path, sep="\t", names=[names[idx]])
         for idx, path in enumerate(paths)],
        axis=1
    )

    # save to csv
    df.to_csv(os.path.join(params.basepath, filename.format(t, 'csv')), sep='\t', index=False)

print(df.sample(10))

