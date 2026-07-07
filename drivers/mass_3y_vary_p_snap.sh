#!/bin/sh
# Vary growth rate from 1% to 10%
# 3 years simulation
# 1 thread

mkdir -p output
mkdir -p errors

OUTPUT=$(readlink -f ./output)/
ERRORS=$(readlink -f ./errors)/

NUM_THREADS=16
NUM_CYCLES=3

SAME_YEAR_PROPORTION="0.12"
FULLY_RANDOM_CITATIONS="0.05"
LOG_LEVEL="1"

growth_percents="1 3 6"
BFS_BATCH_SIZE=25
OUTDEGREE_BAG="tcen_at_least_five"

datasets=("amz" "eu" "yutb" "twt")

mkdir -p ./output/${NUM_CYCLES}y

for GROWTH_PERCENT in 1 3 6
do
    GROWTH_RATE=$(echo "scale=2; $GROWTH_PERCENT/100" | bc)

    for dataset in "${datasets[@]}"
    do
        INPUT_EDGELIST="./seeds/${dataset}_edgelist"
        INPUT_NODELIST="./seeds/${dataset}_nodelist"
        RECENCY_PROBABILITIES="./seeds/${dataset}_recprob"

     
        if [ "$GROWTH_PERCENT" -eq 5 ]; then
            BFS_BATCH_SIZE=50
            echo "For GROWTH_PERCENT = $GROWTH_PERCENT %, BFS_BATCH_SIZE = $BFS_BATCH_SIZE."
        elif [ "$GROWTH_PERCENT" -ge 6 ]; then
            BFS_BATCH_SIZE=60
            echo "For GROWTH_PERCENT = $GROWTH_PERCENT %, BFS_BATCH_SIZE = $BFS_BATCH_SIZE."
        fi

        OUTPUT_FILE="./output/${NUM_CYCLES}y/mass_cuda-${dataset}-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.edgelist"
        OUTPUT_LOG="./output/${NUM_CYCLES}y/mass_cuda-${dataset}-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.log"
        OUTPUT_AUX="./output/${NUM_CYCLES}y/mass_cuda-${dataset}-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.aux"
        OUT_FILE="./output/${NUM_CYCLES}y/mass_cuda-${dataset}-static-${NUM_CYCLES}y-${GROWTH_PERCENT}p-abm.out"

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
done
