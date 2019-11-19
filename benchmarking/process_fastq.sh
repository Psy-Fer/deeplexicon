#!/bin/bash 

####################################
# Pipeline for accuracy evaluation

############ INPUT ################# 
# Example:
# rep3_dmux.tsv        # Deeplicon output. Source:  
# rep3.fastq          # from kyle01@129.94.120.250:/var/services/homes/kyle01/Projects/2019/September/GDDN031789/
#
# Rep3 data was extracted using: 
# rsync --progress --rsh='ssh -T -o Compression=no -x -p2322' kyle01@129.94.120.250:/var/services/homes/kyle01/Projects/2019/September/GDDN031789/*fastq* ./
# find ./GDDN031789 -name "*fastq" -exec cat {} >> rep3.fastq \;

########################################################################
#          We assume the following barcoding order (sequins only): 
#          R2_63_3 -> BC1 
#          R1_81_2 - < BC2
#          R1_103_1 -> BC3 
#          R2_117_1 -> BC4
#

########################################################################
#             Filter sequencing data for benchmarking
########################################################################
# Extract data from server
# pwd : /share/ScratchGeneral/marsmi/deeplexicon/

if [[ $# == 0 ]]; then
	echo -e "\nUSAGE: ./process_fastq.sh deeplexicon_output.csv [ reads.fastq ] [ reference.fasta ]"

if [[ -n "$1" && -e $1 ]]; then
	DMUX=$1
	echo "Demux output found: ${1}"
else
	echo "Demux output ${1} not supplied/found. Exiting."
	exit 1
fi

if [[ -n "$2" && -e $2 ]]; then
	FASTQ=$2
	echo "Input fastq file set to ${2}"
else
	echo "Input fastq ${2} not supplied/found. WARNING: Will bypass sanity checks."
fi

if [[ -n "$3" && -e $3 ]]; then
	REFA=$3
	echo "Reference fasta file set to ${3}"
else
	if [[ ! -e ./refs/barcoded_seqs_ENO2.fa ]]; then 
		echo "Default reference fasta not found in ./refs/barcoded_seqs_ENO2.fa."
		echo "Consider executing from deeplexicon repository root."
		echo "WARNING: Will bypass sanity checks."
	else
		echo "Input fasta ${3} not supplied. Using ./refs/barcoded_seqs_ENO2.fa as default"
	fi
fi


#Align to reference
qsub -cwd -N mm2_18 -pe smp 12 -l mem_requested=4G,h_vmem=4G -V -S /bin/bash -b y ./scripts/minimap2.sh ./refs/barcoded_seqs_ENO2.fa ./fastq/rep3.fastq

# Get readIDs for uniquely mapping reads
awk '$12 == 60' rep3.fastq.paf | cut -f 1,6 | sort -k 1,1 | cut -f 1 | uniq -c | sort -rn | awk '{ if ($1 < 2 && $1 > 0) print $2}' > rep3-guppy_mm2_MQ60_uniqIDs_ENO2.txt

# Filter reads for high-quality unique mappers
join <( sort -k 1,1 rep3.fastq.paf | awk '$12 == 60' ) <( sort -k 1,1 rep3-guppy_mm2-uniq_IDs_ENO2.txt) | sed 's/ /    /g' > rep3_guppy_mm2_MQ60_uniq_ENO2.paf

#discard all reads that map to ENO2
grep -v ENO2 rep3_guppy_mm2_MQ60_uniq_ENO2.paf > rep3_guppy_mm2_MQ60_uniq.paf 



########################################################################
#             Process Deeplexicon output
########################################################################

# Filter demux output to retain uniquely mapping barcoded reads for benchmakring
join -2 2 <( cut -f 1  rep3_guppy_mm2_MQ60_uniq.paf  | sort ) <( sort -k 2,2 rep3_dmux.tsv ) > rep3_dmux_uniqMapped.tsv

# Extract the demux probabilities from Deeplexicon output
# Grab the readID and the 4 barcode probabilities     
awk 'OFS="\t"{ print $1,$5,$6,$7,$8 }' rep3_dmux_uniqMapped.tsv > rep3_dmux_uniqMapped_trim.tsv
# Split the data into one line per probability
awk 'OFS="\t"{ if (NR>1)for (i=2;i<=NF;i++) {print $1"_"(i-1),$i}}' rep3_dmux_uniqMapped_trim.tsv | tee rep3_dmux_uniqMapped_trim_filt.tsv | head -n 4 
# 00019f1e-c28b-4e58-b734-a7e186e585fa_1    0.360632
# 00019f1e-c28b-4e58-b734-a7e186e585fa_2    0.069125
# 00019f1e-c28b-4e58-b734-a7e186e585fa_3    0.52348
# 00019f1e-c28b-4e58-b734-a7e186e585fa_4    0.046763

# 00019f1e-c28b-4e58-b734-a7e186e585fa_1    0.027894
# 00019f1e-c28b-4e58-b734-a7e186e585fa_2    0.485135
# 00019f1e-c28b-4e58-b734-a7e186e585fa_3    0.017352
# 00019f1e-c28b-4e58-b734-a7e186e585fa_4    0.469619

# Generate the control (expected) set from uniquely mapping reads
# Convert SEQUIN gene names into Barcode IDs
awk 'OFS="\t"{ print $1,$6 }' rep3_guppy_mm2_MQ60_uniq.paf  | sed -e 's/R1_103_1/0\t0\t1\t0/g' -e 's/R2_63_3/1\t0\t0\t0/g' -e 's/R2_117_1/0\t0\t0\t1/g' -e 's/R1_81_2/0\t1\t0\t0/g' > rep3_guppy_mm2_MQ60_uniq_expected.tsv
# Split the data into one line per probability
awk 'OFS="\t"{ if (NR>1)for (i=2;i<=NF;i++) {print $1"_"(i-1),$i}}' rep3_guppy_mm2_MQ60_uniq_expected.tsv | tee rep3_guppy_mm2_MQ60_uniq_expected_filt.tsv | head -n 4
# 00019f1e-c28b-4e58-b734-a7e186e585fa_1    0
# 00019f1e-c28b-4e58-b734-a7e186e585fa_2    1
# 00019f1e-c28b-4e58-b734-a7e186e585fa_3    0
# 00019f1e-c28b-4e58-b734-a7e186e585fa_4    0

# Filter the demux probabilities to retain only the known reads
join <( cut -f 1 rep3_dmux_uniqMapped_trim_filt.tsv ) <( sort rep3_guppy_mm2_MQ60_uniq_expected_filt.tsv ) > rep3_guppy_mm2_MQ60_uniq_expected_filt_demuxed.tsv

# Run R_code.R 

