# Welcome to deeplexicon Docs

**DeePlexiCon** is a tool to demultiplex barcoded nanopore direct RNA sequencing data, as well as train the models to do so.


You can find the code here: **[DeePlexiCon](https://github.com/Psy-Fer/deeplexicon)**.

You can read the pre-print here: [Barcoding and demultiplexing Oxford Nanopore native RNA sequencing reads with deep residual learning](https://www.biorxiv.org/content/10.1101/864322v2)

---

## About DeePlexiCon
DeePlexiCon is a tool to demultiplex barcoded direct RNA sequencing reads from Oxford Nanopore Technologies.
Please note that the software has been tested and validated with a set of 4x20bp barcodes listed below:                                     

- Barcode 1: GGCTTCTTCTTGCTCTTAGG
- Barcode 2: GTGATTCTCGTCTTTCTGCG
- Barcode 3: GTACTTTTCTCTTTGCGCGG
- Barcode 4: GGTCTTCGCTCGGTCTTATT

Please see below further instructions about how to build barcoded direct RNA libraries.

---

## How to BUILD BARCODED LIBRARIES

To build the barcoded libraries, the oligo DNA sequences listed below should be used instead of those coming with the direct RNA sequencing kit (RTA). The barcode is embedded in the oligoA sequence, which will be ligated to the RNA molecule during the library preparation.

These oligos are designed to barcode libraries which have been enriched with oligodT beads (i.e. RNA should have polyA tail to anneal to oligoB). Each oligoA matches an oligoB.

OligoA :

- OligoA_shuffle1: 5'-/5Phos/GGCTTCTTCTTGCTCTTAGGTAGTAGGTTC-3' (same as in ONT RTA):
- OligoA_shuffle2: 5'-/5Phos/GTGATTCTCGTCTTTCTGCGTAGTAGGTTC-3'
- OligoA_shuffle3: 5'-/5Phos/GTACTTTTCTCTTTGCGCGGTAGTAGGTTC-3'
- OligoA_shuffle4: 5'-/5Phos/GGTCTTCGCTCGGTCTTATTTAGTAGGTTC-3'

OligoB:

- OligoB_shuffle1: 5’-GAGGCGAGCGGTCAATTTTCCTAAGAGCAAGAAGAAGCCTTTTTTTTTT-3’  (same as in ONT RTA)
- OligoB_shuffle2: 5’-GAGGCGAGCGGTCAATTTTCGCAGAAAGACGAGAATCACTTTTTTTTTT-3’
- OligoB_shuffle3: 5’-GAGGCGAGCGGTCAATTTTCCGCGCAAAGAGAAAAGTACTTTTTTTTTT-3’
- OligoB_shuffle4: 5’-GAGGCGAGCGGTCAATTTTAATAAGACCGAGCGAAGACCTTTTTTTTTT-3’

---
## How to DEMULTIPLEX using DeePlexiCon

### Step 1: Predict barcodes for each read

    python3 deeplexicon.py dmux -p ~/top/fast5/path/ -f multi -m models/resnet20-final.h5 > output.tsv

### Step 2: Split your base-called fastq data (please note that you can filter your output.tsv based on confidence score if you prefer to increase accuracy at the cost of recovery)

    python3 deeplexicon.py split -i output.tsv -q combined.fastq -o dmux_folder/ -s sample_name

### Notes
Please note, the current algorithm has been trained to demultiplex the 4 barcodes shown above. It will not accurately demultiplex reads if different sequences are used.

----
## How to TRAIN DeePlexiCon with different barcodes

    python deeplexicon.py train --path /fast5/top/path/ --train_truth train.tsv --test_truth test.tsv --val_truth val.tsv

----
## Getting help with DeePlexiCon

If you have any troubles using DeePlexiCon, please leave an [issue](https://github.com/Psy-Fer/deeplexicon/issues) in the [github repo](https://github.com/Psy-Fer/deeplexicon) of DeePlexiCon.


## Citing this work

Martin A. Smith, Tansel Ersavas, James M. Ferguson, Huanle Liu, Morghan C. Lucas, Oguzhan Begik, Lilly Bojarski, Kirston Barton, Eva Maria Novoa. Barcoding and demultiplexing Oxford Nanopore direct RNA sequencing reads with deep residual learning. bioRxiv 2019
