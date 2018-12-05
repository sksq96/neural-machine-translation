## Neural Machine Translation


### Preprocess Data

- Change `"` character to `&quot;` for the `vi` files.

```shell
sed -i "" 's/\"/\&quot;/g' *.vi
```

- Combine both source and targeet languaes in one csv file.

```shell
paste -d"\t" train.tok.vi train.tok.en > train.tok.csv
paste -d"\t" dev.tok.vi dev.tok.en > dev.tok.csv
paste -d"\t" test.tok.vi test.tok.en > test.tok.csv
```


### Train 

```shell
usage: train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--encoder ENCODER] [--d_embed D_EMBED] [--d_hidden D_HIDDEN]
                [--d_fc D_FC] [--n_layers N_LAYERS] [--lr LR]
                [--dp_ratio DP_RATIO] [--gpu GPU] [--basepath BASEPATH]
                [--train_file TRAIN_FILE] [--val_file VAL_FILE]
                [--log_every LOG_EVERY] [--dev_every DEV_EVERY]
                [--experiment EXPERIMENT]

Transformer for Language Translation

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS
  --batch_size BATCH_SIZE
  --encoder ENCODER
  --d_embed D_EMBED
  --d_hidden D_HIDDEN
  --d_fc D_FC
  --n_layers N_LAYERS
  --lr LR
  --dp_ratio DP_RATIO
  --gpu GPU
  --basepath BASEPATH
  --train_file TRAIN_FILE
  --val_file VAL_FILE
  --log_every LOG_EVERY
  --dev_every DEV_EVERY
  --experiment EXPERIMENT
```

