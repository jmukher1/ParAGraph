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

SAME_YEAR_PROPORTION="0.12"
FULLY_RANDOM_CITATIONS="0.05"
LOG_LEVEL="1"

OUTDEGREE_BAG="tcen_at_least_five"

datasets=("amz" "eu" "yutb" "twt")

for GROWTH_PERCENT in 1 3 6
do
    GROWTH_RATE=$(echo "scale=2; $GROWTH_PERCENT/100" | bc)

    for dataset in "${datasets[@]}"
    do
        INPUT_EDGELIST="./seeds/${dataset}_edgelist"
        INPUT_NODELIST="./seeds/${dataset}_nodelist"
        RECENCY_PROBABILITIES="./seeds/${dataset}_recprob"

        OUTPUT_FILE="./output/cpu-parallel-${dataset}-profile-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.edgelist"
        OUTPUT_LOG="./output/cpu-parallel-${dataset}-profile-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p-${NUM_THREADS}t.log"
        OUTPUT_AUX="./output/cpu-parallel-${dataset}-profile-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.aux"
        OUT_FILE="./output/cpu-parallel-${dataset}-profile-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p-${NUM_THREADS}t.out"

        echo "Running dataset=${dataset} growth rate ${GROWTH_PERCENT}% with ${NUM_THREADS} thread"
        echo $OUTPUT_FILE
        echo $OUTPUT_AUX

        time ./cpp_abm \
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
        2>${ERRORS}/abm-${dataset}-${GROWTH_PERCENT}p.err \
        1>${OUT_FILE}

    done
done