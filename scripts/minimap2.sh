#!/bin/bash

module load marsmi/nanopore/minimap2

FQ=$2
FA=$1
ALNPAF=${FQ%*.fastq.gz}.paf
OUTPUT=./aln/${ALNPAF##*/}
THREADS=12


MM2_BIN=$( which minimap2 )
if [[ -z ${MM2_BIN} ]] ; then
  echo "[ERROR] Cannot find 'minimap2' executable. Please link it to your \$PATH and try again"
  exit 1
fi

if [[ ! -d aln ]] ; then
  mkdir aln
fi

echo "running minimap2..."
time $MM2_BIN --secondary=no -k14 -t ${THREADS} ${FA} ${FQ} > ${OUTPUT}

echo "Copying data..."


echo "Done!"

