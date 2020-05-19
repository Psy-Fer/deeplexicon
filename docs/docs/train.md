# Train a custom model

To train a custom model, a dataset with each barcode attached to a read that will uniquly map with regard to all other barcodes is required.

Some examples:

- Synthetic RNA controls [sequins](https://www.sequinstandards.com/) attached to each barcode
- Different species for each barcode

So long as when you map with minimap2, you can group the reads by expected barcode output.

### Input requirements

1. Raw fast5 files
2. Truth table of readID->barcode

The truth table of readIDs should be in one-hot format, ie,

    readID  1   0   0   0
    readID  0   1   0   0
    readID  0   0   1   0
    readID  0   0   0   1

With binary classification of barcode 1, 2, 3, 4 respectively.
Of course, this can be extended to any number of barcodes required, and accross multiple runs.

This should then be split into `training`, `testing`, and `validation` files.

Do this by placing:
- 80% of the reads into a `training` file. `--train_truth`
- 10% of the reads into a `testing` file. `--test_truth`
- 10% of the reads into a `validation` file. `--val_truth`


### Running the training

Training requires a CUDA compatible GPU and the correct libraries installed.

Commence training, with validation, with the following:

    python deeplexicon.py train --path /fast5/top/path/ --train_truth train.tsv --test_truth test.tsv --val_truth val.tsv



### Full description

    train.add_argument('-p', '--path', nargs='+',
                        help="Input path(s) of all used fast5s")
    train.add_argument('-t', '--train_truth', nargs='+',
                        help="Traiing truth set(s) in one-hot format eg: readID, 0, 0, 1, 0 for barcode 3 of 4 ")
    train.add_argument('-s', '--test_truth', nargs='+',
                        help="Testing truth set(s) in one-hot format eg: readID, 0, 0, 1, 0 for barcode 3 of 4 ")
    train.add_argument('-u', '--val_truth', nargs='+',
                        help="Validation truth set(s) in one-hot format eg: readID, 0, 0, 1, 0 for barcode 3 of 4 ")
    train.add_argument('-n', '--network', default="ResNet20",
                        help="Network to use (see table in docs)")
    train.add_argument('--net_version', type=int, default=2,
                        help="Network version to use (see table in docs)")
    train.add_argument('-e', '--epochs', type=int, default=40,
                        help="epochs to run")
    train.add_argument('-v', '--verbose', action='count', default=0,
                        help="Verbose output [v/vv/vvv]")
