#!/bin/sh

mkdir -p output
mkdir -p errors

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

# Set BFS batch size (based on your logic)
if [ "$GROWTH_PERCENT" -eq 5 ]; then
    BFS_BATCH_SIZE=20
elif [ "$GROWTH_PERCENT" -ge 6 ]; then
    BFS_BATCH_SIZE=30
else
    BFS_BATCH_SIZE=10
fi

echo "Using GROWTH_PERCENT = $GROWTH_PERCENT %, BFS_BATCH_SIZE = $BFS_BATCH_SIZE"

# Run 10 independent simulations
for RUN_ID in $(seq 1 10)
do
    echo "=============================="
    echo "Run $RUN_ID / 10"
    echo "=============================="

    OUTPUT_FILE="./output/mass_cuda-run${RUN_ID}-${NUM_CYCLES}y-${GROWTH_PERCENT}p.edgelist"
    OUTPUT_AUX="./output/mass_cuda-run${RUN_ID}-${NUM_CYCLES}y-${GROWTH_PERCENT}p.aux"
    OUT_FILE="./output/mass_cuda-run${RUN_ID}-${NUM_CYCLES}y-${GROWTH_PERCENT}p.out"
    ERR_FILE="./errors/mass_cuda-run${RUN_ID}-${NUM_CYCLES}y-${GROWTH_PERCENT}p.err"

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
    --auxiliary-information-file ${OUTPUT_AUX} \
    > ${OUT_FILE} 2> ${ERR_FILE}

done