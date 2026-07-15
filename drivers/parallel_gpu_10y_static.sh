#!/bin/sh
# Vary growth rate from 1% to 10%
# 10 years simulation
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
USE_WARP_BFS="true"
USE_BATCHING="true"
MAX_BATCH_SIZE="100000"
USE_MULTISTAGE_KERNEL="true" 

for GROWTH_PERCENT in 1 3 6
do
    GROWTH_RATE=$(echo "scale=2; $GROWTH_PERCENT/100" | bc)

    OUTPUT_FILE="./output/baseline-gpu-parallel-static-${NUM_CYCLES}y-${GROWTH_PERCENT}p.edgelist"
    OUTPUT_LOG="./output/baseline-gpu-parallel-static-${NUM_CYCLES}y-${GROWTH_PERCENT}p-${NUM_THREADS}t.log"
    OUTPUT_AUX="./output/baseline-gpu-parallel-static-${NUM_CYCLES}y-${GROWTH_PERCENT}p.aux"
    OUT_FILE="./output/baseline-gpu-parallel-static-${NUM_CYCLES}y-${GROWTH_PERCENT}p-${NUM_THREADS}t.out"

    echo "Running growth rate ${GROWTH_PERCENT}% with ${NUM_THREADS} thread"
    echo $OUTPUT_FILE
    echo $OUTPUT_AUX

    ./abm \
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
    --use-warp-bfs ${USE_WARP_BFS} \
    --use-batching ${USE_BATCHING} \
    --max-batch-size ${MAX_BATCH_SIZE} \
    --use-multistage-kernel ${USE_MULTISTAGE_KERNEL} \
    --num-processors ${NUM_THREADS} \
    --log-level ${LOG_LEVEL} \
    2>${ERRORS}/abm-${GROWTH_PERCENT}p.err \
    1>${OUT_FILE}

done