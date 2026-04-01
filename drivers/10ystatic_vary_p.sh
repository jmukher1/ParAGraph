#!/bin/sh
# Vary growth rate from 1% to 10%
# 3 years simulation
# 1 thread

mkdir -p output
mkdir -p errors

OUTPUT=$(readlink -f ./output)/
ERRORS=$(readlink -f ./errors)/

NUM_THREADS=16
NUM_CYCLES=10

INPUT_EDGELIST="sj_edgelist"
INPUT_NODELIST="sj_nodelist"
OUTDEGREE_BAG="tcen_at_least_five"
RECENCY_PROBABILITIES="sj_recprob"

SAME_YEAR_PROPORTION="0.12"
FULLY_RANDOM_CITATIONS="0.05"
LOG_LEVEL="1"

mkdir -p output/${NUM_CYCLES}y
#growth_percents="1 3 5 7 9 11 15 20 25 50 75 100"
growth_percents="1 3 5 6"

mkdir -p output/${NUM_CYCLES}y

for GROWTH_PERCENT in $growth_percents
do
    GROWTH_RATE=$(echo "scale=2; $GROWTH_PERCENT/100" | bc)

    OUTPUT_FILE="./output/${NUM_CYCLES}y/gpu-opt-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.edgelist"
    OUTPUT_LOG="./output/${NUM_CYCLES}y/gpu-opt-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.log"
    OUTPUT_AUX="./output/${NUM_CYCLES}y/gpu-opt-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.aux"
    OUT_FILE="./output/${NUM_CYCLES}y/gpu-opt-static-${NUM_CYCLES}y-${GROWTH_PERCENT}p-abm.out"

    echo "Running growth rate ${GROWTH_PERCENT}% for ${NUM_CYCLES} years."
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
