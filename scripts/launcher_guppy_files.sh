#!/bin/bash

NUM_JOBS=$(find . -type f -name "*.fast5" | wc -l)

qsub -t 1-${NUM_JOBS} ./guppy_basecall_files.sge

#AJOBID=$( $QCMD ) # keep job id

#qsub -hold_jid $AJOBID "./finisher.sh"
