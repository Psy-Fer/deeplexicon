#!/bin/bash
#$ -cwd
#$ -N FINISH
#$ -S /bin/bash
#$ -b y
#$ -j y 
#$ -o logs/FINISHER.log

INPUT=${1%*/}
FASTQDIR=./fastq/${INPUT##*/}_guppy-3.2.4

find ${FASTQDIR} -name "*fastq" -exec cat {} \; > ${FASTQDIR}.fastq

( head -n 1 ${FASTQDIR}/1/sequencing_summary.txt ; find ${FASTQDIR} -name "*_summary.txt" -exec cat {} \; | grep -v filename ) > ./seq_sums/${1##*/}_guppy-3.2.4_seqsum.txt

rm -rf ${FASTQDIR} 

