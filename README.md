## Neural Machine Translation


### Preprocess Data

- Change `"` character to `&quot;` for the `vi` files.

```
sed -i "" 's/\"/\&quot;/g' *.vi
```

- Combine both source and targeet languaes in one csv file.

```
paste -d"\t" train.tok.vi train.tok.en > train.tok.csv
paste -d"\t" dev.tok.vi dev.tok.en > dev.tok.csv1
paste -d"\t" test.tok.vi test.tok.en > test.tok.csv
```
