#!/bin/sh
# Vary growth rate from 1% to 10%
# 3 years simulation
# 1 thread

mkdir -p output
mkdir -p errors

OUTPUT=$(readlink -f ./output)/
ERRORS=$(readlink -f ./errors)/

NUM_THREADS=16
NUM_CYCLES=30

INPUT_EDGELIST="sj_edgelist"
INPUT_NODELIST="sj_nodelist"
OUTDEGREE_BAG="tcen_at_least_five"
RECENCY_PROBABILITIES="sj_recprob"

SAME_YEAR_PROPORTION="0.12"
FULLY_RANDOM_CITATIONS="0.05"
LOG_LEVEL="1"

for GROWTH_PERCENT in 1 5
do
    GROWTH_RATE=$(echo "scale=2; $GROWTH_PERCENT/100" | bc)

    OUTPUT_FILE="./output/cpu-parallel-profile-static-output-1p5p-${NUM_CYCLES}y-${GROWTH_PERCENT}p.edgelist"
    OUTPUT_LOG="./output/cpu-parallel-profile-static-output-1p5p-${NUM_CYCLES}y-${GROWTH_PERCENT}p-${NUM_THREADS}t.log"
    OUTPUT_AUX="./output/cpu-parallel-profile-static-output-1p5p-${NUM_CYCLES}y-${GROWTH_PERCENT}p.aux"
    OUT_FILE="./output/cpu-parallel-profile-static-output-1p5p-${NUM_CYCLES}y-${GROWTH_PERCENT}p-${NUM_THREADS}t.out"

    echo "Running growth rate ${GROWTH_PERCENT}% with ${NUM_THREADS} thread"
    echo $OUTPUT_FILE
    echo $OUTPUT_AUX

    time ./abm \
    --edgelist ${INPUT_EDGELIST} \
    --nodelist ${INPUT_NODELIST} \
    --out-degree-bag ${OUTDEGREE_BAG} \
    --recency-probabilities ${RECENCY_PROBABILITIES} \
    --alpha 0.5 \
    --preferential-weight 0.33 \
    --recency-weight 0.33 \
    --fitness-weight 0.33 \
    --growth-rate ${GROWTH_RATE} \
    --fully-random-citations ${FULLY_RANDOM_CITATIONS} \
    --num-cycles ${NUM_CYCLES} \
    --same-year-proportion ${SAME_YEAR_PROPORTION} \
    --output-file ${OUTPUT_FILE} \
    --auxiliary-information-file ${OUTPUT_AUX} \
    --log-file ${OUTPUT_LOG} \
    --num-processors ${NUM_THREADS} \
    --log-level ${LOG_LEVEL} \
    2>${ERRORS}/abm-${GROWTH_PERCENT}p.err \
    1>${OUT_FILE}

done