#!/bin/bash
#$ -cwd
#$ -N FINISH
#$ -S /bin/bash
#$ -b y

#NUM_JOBS=$(find . -type f -name "*.fast5" | wc -l)
NUM_ERROR=$(for f in guppy-3.1.5/*/*/*.txt; do nice grep "ERROR" $f; done | wc -l)
NUM_SUCC=$(for f in guppy-3.1.5/*/*/*.txt; do nice grep "succ" $f; done | wc -l)
 
if [ $NUM_ERROR -gt 0 ]; then echo "CUDA ERROR: At least this many jobs failed: "$NUM_ERROR; exit 1; fi
#if [ ! $NUM_SUCC -eq $NUM_JOBS ]; then echo "MISSING JOBS: "$NUM_SUCC" successful jobs, but total is "$NUM_JOBS; exit 1; fi

#cat guppy-3.1.5/*/*/seq*.txt > raw.txt && awk ' /^filename/ && FNR > 1 {next} {print $0} ' raw.txt > sequencing_summary.txt && rm raw.txt
#cat guppy-3.1.5/*/*/*.fastq > HAC.fastq && gzip HAC.fastq
