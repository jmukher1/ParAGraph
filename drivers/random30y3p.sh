#!/bin/bash

# 30 years 3% growth all random 

mkdir -p output
mkdir -p errors
OUTPUT=$(readlink -f ./output)/
ERRORS=$(readlink -f ./errors)/

NUM_THREADS=1
GROWTH_RATE=0.03
NUM_CYCLES=30

GROWTH_PERCENT=$(echo "$GROWTH_RATE * 100" | bc | cut -d. -f1)
OUTPUT_FILE="./output/gpu-random-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.edgelist"
OUTPUT_LOG="./output/gpu-random-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.log"
OUTPUT_AUX="./output/gpu-random-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.aux"
CMDLINEOUT_FILE="./output/gpu-random-${NUM_CYCLES}y-${GROWTH_PERCENT}p-abm.out"

echo $OUTPUT_FILE
echo $OUTPUT_AUX

INPUT_EDGELIST="sj_edgelist"
INPUT_NODELIST="sj_nodelist"
OUTDEGREE_BAG="tcen_at_least_five"
RECENCY_PROBABILITIES="sj_recprob"
SAME_YEAR_PROPORTION="0.12"
FULLY_RANDOM_CITATIONS="0.05"
LOG_LEVEL="1"


/usr/bin/time -v ./abm --edgelist ${INPUT_EDGELIST} --nodelist ${INPUT_NODELIST} --out-degree-bag ${OUTDEGREE_BAG} --recency-probabilities ${RECENCY_PROBABILITIES} --growth-rate ${GROWTH_RATE} --fully-random-citations ${FULLY_RANDOM_CITATIONS} --num-cycles ${NUM_CYCLES} --same-year-proportion ${SAME_YEAR_PROPORTION} --output-file ${OUTPUT_FILE} --auxiliary-information-file ${OUTPUT_AUX} --log-file ${OUTPUT_LOG} --num-processors ${NUM_THREADS} --log-level ${LOG_LEVEL} 2>${ERRORS}/abm.err 1>${CMDLINEOUT_FILE}
