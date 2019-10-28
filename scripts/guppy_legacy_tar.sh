#!/bin/bash
#$ -cwd
#$ -N GPU_GUP
#$ -S /bin/bash
#$ -b y
#$ -l mem_requested=20G
#$ -l h_vmem=20G
#$ -l tmp_requested=5G
#$ -l nvgpu=1
#$ -pe smp 4
#$ -o /dev/null
#$ -e /dev/null

# Models files found here:
# /share/ClusterShare/software/contrib/shacar/ont-guppy/

IN_DIR=${1}
OUT_DIR=${2}

if [ -z $IN_DIR ]; then echo "Edit -t parameter, IN_DIR and OUT_DIR required"; exit 1; fi

READ_TAR=$(ls -v ${IN_DIR}/*.tar | sed -n ${SGE_TASK_ID}p)
TAR_FILE=${READ_TAR##*/}
BATCH=$(basename "${READ_TAR}" .tar)
OUTPUT=${OUT_DIR}/${BATCH}
CONFIG=/share/ClusterShare/software/contrib/shacar/ont-guppy-3.1.5/data/rna_r9.4.1_70bps_hac.cfg

if [ ! -d ${OUTPUT} ]; then mkdir -p ${OUTPUT}; fi

echo "TMPDIR="${TMPDIR}
mkdir ${TMPDIR}/fast5
tar -C ${TMPDIR}/fast5 -xf ${READ_TAR} --wildcards --no-anchored '*.fast5'

module load shacar/ont-guppy/3.1.5 

time guppy_basecaller --chunks_per_runner 1500 --gpu_runners_per_device 1 --cpu_threads_per_caller 4 -x "cuda:0 cuda:1 cuda:2 cuda:3" -r -i ${TMPDIR}/fast5 -c ${CONFIG} -s ${OUTPUT} >${OUTPUT}/time.txt 2>&1

echo $SECONDS >> basecalling_runtime_seconds.txt
