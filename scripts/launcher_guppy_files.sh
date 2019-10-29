#!/bin/bash

INPUT=${1%*/} #chop off last slash

NUM_JOBS=$(find ${1} -type f -name "*.fast5" | wc -l)

CMD="qsub -sync y -t 1-${NUM_JOBS} ./scripts/guppy_basecall_files.sge ${INPUT}"

echo $CMD 

WAIT4IT=$( $CMD )

qsub -hold_jid $WAIT4IT ./scripts/finisher.sh ${INPUT}
