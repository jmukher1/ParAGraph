#!/bin/sh
# ER Probability from 0.01% to 0.1%
# 3 - 10 years simulation
# 16 thread

mkdir -p output
mkdir -p errors

OUTPUT=$(readlink -f ./output)/
ERRORS=$(readlink -f ./errors)/

NUM_THREADS=16
INPUT_EDGELIST="sj_edgelist"
INPUT_NODELIST="sj_nodelist"
OUTDEGREE_BAG="tcen_at_least_five"
RECENCY_PROBABILITIES="sj_recprob"

SAME_YEAR_PROPORTION="0.12"
FULLY_RANDOM_CITATIONS="0.05"
LOG_LEVEL="1"
MODEL="er-gnp"

ER_PROBABILITY="0.0001"
ER_PROBABILITY_LABEL=$(awk "BEGIN {printf \"%.2f\", 100 * $ER_PROBABILITY}")
mkdir -p output/${NUM_CYCLES}y/${ER_PROBABILITY_LABEL}

for GROWTH_PERCENT in 1 3 6
do
    GROWTH_RATE=$(echo "scale=4; $GROWTH_PERCENT/100" | bc -l)
    GROWTH_LABEL=$(echo "$GROWTH_PERCENT" | sed 's/0\./dot/; s/\./dot/') 

    for NUM_CYCLES in 3 10 30
    do
        mkdir -p output/${NUM_CYCLES}y/${ER_PROBABILITY_LABEL}

        OUTPUT_FILE="./output/${NUM_CYCLES}y/${ER_PROBABILITY_LABEL}/gpu-er-${dataset}-static-model-${MODEL}-${NUM_CYCLES}y-${GROWTH_LABEL}p-${ER_PROBABILITY_LABEL}p.edgelist"
        OUTPUT_LOG="./output/${NUM_CYCLES}y/${ER_PROBABILITY_LABEL}/gpu-er-${dataset}-static-model-${MODEL}-${NUM_CYCLES}y-${GROWTH_LABEL}p-${ER_PROBABILITY_LABEL}p-${NUM_THREADS}t.log"
        OUTPUT_AUX="./output/${NUM_CYCLES}y/${ER_PROBABILITY_LABEL}/gpu-er-${dataset}-static-model-${MODEL}-${NUM_CYCLES}y-${GROWTH_LABEL}p-${ER_PROBABILITY_LABEL}p.aux"
        OUT_FILE="./output/${NUM_CYCLES}y/${ER_PROBABILITY_LABEL}/gpu-er-${dataset}-static-model-${MODEL}-${NUM_CYCLES}y-${GROWTH_LABEL}p-${ER_PROBABILITY_LABEL}p-${NUM_THREADS}t.out"

        echo "Running growth rate = ${GROWTH_RATE} = ${GROWTH_PERCENT}% with ${NUM_THREADS} thread"
        echo $OUTPUT_FILE
        echo $OUTPUT_AUX

        time ./abm_er \
        --model er-gnp \
        --er-probability ${ER_PROBABILITY} \
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
done 