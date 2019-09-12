This section contains methodological details and data processing information, as described in the associated manuscript. 

1. DATASETS

There are 3 native (direct) RNA sequencing datasets related to this work : 

  A. Rep1: 4 barcodes ligated to 4 unique in-vitro transcribed RNA templates. 
  
  B. Rep2: The same 4 barcodes ligated to 8 unique in-vitro transcribed RNA templates, 2 per barcode. 
  
  C. Rep3: The same 4 barcodes ligated to 4 unique in-vitro transcribed RNA templates AND a transcript common to all 4 samples, the S. cerevisiae ENO2 gene, which consists of the experimental control sequence included with the ONT SQK-RNA002 library preparation kit. 

All 3 datasets were sequenced on a Oxford Nanopore MinION/GridION 9.4.1 flowcell. Base-calling was performed with the ONT guppy software. 

#more details on file structures

2. DATA PROCESSING

2.1 MAPPING

Rep1 and Rep2 were used for neural net training. In particular, the raw Fast5 files were base-called to fastq and aligned to a fasta file containing the reference sequences in question. For Rep1, this contained the 4 unique sequences, while Rep2 was mapped to a fasta file with the 8 unique sequences. Mapping was performed with Minimap2 (see manuscript for details). 

2.2 FILTERING

For stringency, only reads with MAPQ60 were retained. Furthermore, only reads that mapped uniquely to their respectively ligated reference sequence were retained.
Read IDs corresponding to the reads that passed these filtering criteria were retained and used to extract the corresponding raw signal data in fast5 files using the Fast5_fetcher script from SquiggleKit.
These reads were then pooled according to their respective barcodes. 

2.3 EXTRACTION

2.3.1 Fast5_fetcher

2.3.2 SquigglePull 

2.4 SEGMENTATION

The output of SquigglePull was processed with a modified version of Segmenter from SquiggleKit, which was tuned fpr the task of identifying DNA adapter signal from native poly(A) RNA nanopore sequencing data. 

2.5 SPLITTING

The extracted barcode signals were randomly organised into 3 sets of 4 groups for training and testing as follows:

Set 1: Training set composed of 40,000 reads for each group, used to build/refine the neural net.

Set 2: Testing set composed of 10,000 reads for each group, used to evaluate/score the neural net

Set 3: Witheld testing set (validation set) composed of 10,000 reads for each group. 

3. TRAINING


4. TESTING

Sets 2 and 3 were used to assess the accuracy of the resulting neual net classifier. The clasiifer emits a likelihood percentage for each barcode, which was used for evaluating accuracy. 
Accuracy was assessed in R using the ROCit package. The average Optimal Youden Index for both sets was used to determine the optimal cutoff from ROC curves.  
The final accuracy metrics were determined for this optimal cutoff. More stringent cutoffs were determined by extrcting the cut-offs associated with lower false-positive metrics. 





