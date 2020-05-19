#How to BUILD BARCODED LIBRARIES

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


<!-- TODO: Link to protocol -->
