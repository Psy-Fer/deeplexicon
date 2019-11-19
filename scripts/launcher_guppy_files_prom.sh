#!/bin/bash

INPUT=${1%*/} #chop off last slash

NUM_JOBS=$(find ${1} -type f -name "*.fast5" | wc -l)

CMD="qsub -t 1-${NUM_JOBS} ./scripts/guppy_basecall_files_prom.sge ${INPUT}"

echo $CMD && $CMD

#AJOBID=$( $QCMD ) # keep job id

#qsub -hold_jid $AJOBID "./finisher.sh"
