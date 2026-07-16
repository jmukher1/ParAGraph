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

growth_percents="1 3 6"
BFS_BATCH_SIZE=40

mkdir -p ./output/${NUM_CYCLES}y


for GROWTH_PERCENT in $growth_percents
do
    GROWTH_RATE=$(echo "scale=2; $GROWTH_PERCENT/100" | bc)
     
    if [ "$GROWTH_PERCENT" -ge 6 ]; then
        BFS_BATCH_SIZE=50
        echo "For GROWTH_PERCENT = $GROWTH_PERCENT %, BFS_BATCH_SIZE = $BFS_BATCH_SIZE."
    fi

    OUTPUT_FILE="./output/${NUM_CYCLES}y/mass_cuda-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.edgelist"
    OUTPUT_LOG="./output/${NUM_CYCLES}y/mass_cuda-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.log"
    OUTPUT_AUX="./output/${NUM_CYCLES}y/mass_cuda-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.aux"
    OUT_FILE="./output/${NUM_CYCLES}y/mass_cuda-static-${NUM_CYCLES}y-${GROWTH_PERCENT}p-abm.out"

    echo "Running growth rate ${GROWTH_PERCENT}% for ${NUM_CYCLES} years."
    echo $OUTPUT_FILE
    echo $OUTPUT_AUX

    time ./build/bin/citation_abm_mass \
    --edgelist ${INPUT_EDGELIST} \
    --nodelist ${INPUT_NODELIST} \
    --out-degree-bag ${OUTDEGREE_BAG} \
    --recency-probabilities ${RECENCY_PROBABILITIES} \
    --bfs-batches ${BFS_BATCH_SIZE} \
    --alpha 0.5 \
    --preferential-weight 0.33 \
    --recency-weight 0.33 \
    --fitness-weight 0.33 \
    --growth-rate ${GROWTH_RATE} \
    --fully-random-citations ${FULLY_RANDOM_CITATIONS} \
    --num-cycles ${NUM_CYCLES} \
    --same-year-proportion ${SAME_YEAR_PROPORTION} \
    --output-file ${OUTPUT_FILE} \
    --auxiliary-information-file ${OUTPUT_AUX} | tee ${OUT_FILE}
    #--log-file ${OUTPUT_LOG} \
    #--num-processors ${NUM_THREADS} \
    #--log-level ${LOG_LEVEL} | 
    #2>${ERRORS}/abm-${GROWTH_PERCENT}p.err \
    #1>${OUT_FILE}

done
