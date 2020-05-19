# Split reads

Splitting fastq reads using the `dmux` output file
<!-- , or splitting mapped reads for use in `train` -->

### Split basecalled fastq data

please note that you can filter your output.tsv based on confidence score if you prefer to increase accuracy at the cost of recovery

    python3 deeplexicon.py split -i output.tsv -q combined.fastq -o dmux_folder/ -s sample_name
<!--
### Split mapped reads (.paf) for use in training

    python3 deeplexicon.py split --train -i mapped.paf -o split_train_output/ -->
