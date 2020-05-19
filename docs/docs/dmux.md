# How to DEMULTIPLEX using DeePlexiCon

### Step 1: Predict barcodes for each read

    python3 deeplexicon.py dmux -p ~/top/fast5/path/ -f multi -m models/resnet20-final.h5 > output.tsv

### Step 2: Split your base-called fastq data

(please note that you can filter your output.tsv based on confidence score if you prefer to increase accuracy at the cost of recovery)

    python3 deeplexicon.py split -i output.tsv -q combined.fastq -o dmux_folder/ -s sample_name

### Notes
- Please note, the current algorithm has been trained to demultiplex the 4 barcodes shown above.
- It will not accurately demultiplex reads if different sequences are used.
- Any mix of the 4 currently trained barcodes will still work
- For more barcodes, see training section for more information
