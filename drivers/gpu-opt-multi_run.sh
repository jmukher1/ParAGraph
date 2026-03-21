#!/bin/sh

mkdir -p output 

NUM_THREADS=1
NUM_CYCLES=3

INPUT_EDGELIST="sj_edgelist"
INPUT_NODELIST="sj_nodelist"
OUTDEGREE_BAG="tcen_at_least_five"
RECENCY_PROBABILITIES="sj_recprob"

SAME_YEAR_PROPORTION="0.12"
FULLY_RANDOM_CITATIONS="0.05"
LOG_LEVEL="1"

# Fixed growth = 3%
GROWTH_PERCENT=3
GROWTH_RATE=$(echo "scale=2; $GROWTH_PERCENT/100" | bc) 

echo "Using GROWTH_PERCENT = $GROWTH_PERCENT %, BFS_BATCH_SIZE = $BFS_BATCH_SIZE"

# Run 10 independent simulations
for RUN_ID in $(seq 1 10)
do
    echo "=============================="
    echo "Run $RUN_ID / 10"
    echo "=============================="

    OUTPUT_FILE="./output/gpu-opt-run${RUN_ID}-${NUM_CYCLES}y-${GROWTH_PERCENT}p.edgelist"
    OUTPUT_AUX="./output/gpu-opt-run${RUN_ID}-${NUM_CYCLES}y-${GROWTH_PERCENT}p.aux"
    OUT_FILE="./output/gpu-opt-run${RUN_ID}-${NUM_CYCLES}y-${GROWTH_PERCENT}p.out" 
    LOG_FILE="./output/gpu-opt-run${RUN_ID}-${NUM_CYCLES}y-${GROWTH_PERCENT}p.log" 

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
    --log-file ${LOG_FILE} --log-level 1 | tee ${OUT_FILE}  

done