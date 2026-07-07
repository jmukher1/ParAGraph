#!/bin/sh
# ER Probability sweep: 0.01% to 0.1%
# Growth rate sweep: 1%, 3%, 6%
# Simulation length sweep: 3, 10, 30 years
# 16 threads

mkdir -p output
mkdir -p errors

OUTPUT=$(readlink -f ./output)/
ERRORS=$(readlink -f ./errors)/

NUM_THREADS=16

OUTDEGREE_BAG="tcen_at_least_five"

SAME_YEAR_PROPORTION="0.12"
FULLY_RANDOM_CITATIONS="0.05"
LOG_LEVEL="1"
MODEL="er-gnp"

datasets=("sj")
er_probabilities=(0.0001 0.0003 0.0005 0.001)
BFS_BATCH_SIZE=10

for GROWTH_PERCENT in 1 3 6
do
    GROWTH_RATE=$(echo "scale=4; $GROWTH_PERCENT/100" | bc -l)
    GROWTH_LABEL=$(echo "$GROWTH_PERCENT" | sed 's/0\./dot/; s/\./dot/')

    # Set BFS batch size (based on your logic)
    if [ "$GROWTH_PERCENT" -eq 5 ]; then
        BFS_BATCH_SIZE=20
    elif [ "$GROWTH_PERCENT" -ge 6 ]; then
        BFS_BATCH_SIZE=30
    else
        BFS_BATCH_SIZE=10
    fi

    for NUM_CYCLES in 3 10 30
    do
        for ER_PROBABILITY in "${er_probabilities[@]}"
        do
            ER_PROBABILITY_LABEL=$(awk "BEGIN {printf \"%.2f\", 100 * $ER_PROBABILITY}")
            mkdir -p output/${NUM_CYCLES}y/${ER_PROBABILITY_LABEL}

            for dataset in "${datasets[@]}"
            do
                INPUT_EDGELIST="./seeds/${dataset}_edgelist"
                INPUT_NODELIST="./seeds/${dataset}_nodelist"
                RECENCY_PROBABILITIES="./seeds/${dataset}_recprob"

                OUTPUT_FILE="./output/${NUM_CYCLES}y/${ER_PROBABILITY_LABEL}/mass-${MODEL}_cuda-${dataset}-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.edgelist"
                OUTPUT_LOG="./output/${NUM_CYCLES}y/${ER_PROBABILITY_LABEL}/mass-${MODEL}_cuda-${dataset}-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.log"
                OUTPUT_AUX="./output/${NUM_CYCLES}y/${ER_PROBABILITY_LABEL}/mass-${MODEL}_cuda-${dataset}-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.aux"
                OUT_FILE="./output/${NUM_CYCLES}y/${ER_PROBABILITY_LABEL}/mass-${MODEL}_cuda-${dataset}-static-${NUM_CYCLES}y-${GROWTH_PERCENT}p-abm.out"

                echo "Running dataset=${dataset} growth=${GROWTH_PERCENT}% cycles=${NUM_CYCLES}y er_prob=${ER_PROBABILITY} (${ER_PROBABILITY_LABEL}%) threads=${NUM_THREADS}"
                echo $OUTPUT_FILE
                echo $OUTPUT_AUX 
                time ./build/bin/citation_abm_mass \
                    --model er-gnp \
                --er-probability ${ER_PROBABILITY} \
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
            done
        done
    done
done

 