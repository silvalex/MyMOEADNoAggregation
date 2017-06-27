#!/bin/sh

NUM_RUNS=50

for i in {1..8}; do
  qsub -t 1-$NUM_RUNS:1 mymoead.sh ~/workspace/wsc2008/Set0${i}MetaData 2008-moead${i} moead.params;
done
