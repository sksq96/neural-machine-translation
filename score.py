import os
from argparse import ArgumentParser

import sacrebleu

parser = ArgumentParser(description='Transformer for Language Translation')
parser.add_argument('--source_file', type=str, default='iwslt-vi-en/test.tok.en')
parser.add_argument('--generated_file', type=str, default='iwslt-vi-en/generated.test.tok.en')
params = parser.parse_args()


if __name__ == "__main__":
    f1 = open(params.generated_file).read()
    f2 = open(params.source_file).read()
    print(sacrebleu.corpus_bleu(f1, f2))

