#!/bin/bash
#$ -cwd
#$ -N MM2
#$ -S /bin/bash
#$ -b y
#$ -j y
#$ -l mem_requested=2G
#$ -l h_vmem=2G
#$ -l tmp_requested=50M
#$ -pe smp 12
#$ -o logs/mapping.sge.log
module load marsmi/nanopore/minimap2

FQ=$2
FA=$1
ALNPAF=${FQ%*.fastq}.paf
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

echo "running minimap2, version :"
$MM2_BIN -v

time $MM2_BIN --secondary=no -k14 -t ${THREADS} ${FA} ${FQ} > ${OUTPUT}

# Get readIDs for uniquely mapping reads
UNIQ_ID=${OUTPUT%*.paf}_MQ60_uniqIDs.txt
awk '$12 == 60' ${OUTPUT}  | cut -f 1,6 | sort -k 1,1 | cut -f 1 | uniq -c | sort -rn | awk '{ if ($1 < 2 && $1 > 0) print $2}' > ${UNIQ_ID}

# Filter reads for high-quality unique mappers
join <( sort -k 1,1 ${OUTPUT} | awk '$12 == 60' ) <( sort ${UNIQ_ID} ) | sed -e 's/ /\t/g' > ${OUTPUT%*.paf}_MQ60_uniq.paf

echo "Done!"

