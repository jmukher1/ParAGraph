#!/bin/sh
# 30 years 3% growth - static


mkdir -p output
mkdir -p errors
OUTPUT=$(readlink -f ./output)/
ERRORS=$(readlink -f ./errors)/

NUM_THREADS=64
GROWTH_RATE=0.03
NUM_CYCLES=10

GROWTH_PERCENT=$(echo "$GROWTH_RATE * 100" | bc | cut -d. -f1)
OUTPUT_FILE="./output/gpu-opt-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.edgelist"
OUTPUT_LOG="./output/gpu-opt-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.log"
OUTPUT_AUX="./output/gpu-opt-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.aux"
CMDLINEOUT_FILE="./output/gpu-opt-static-${NUM_CYCLES}y-${GROWTH_PERCENT}p-abm.out"

echo $OUTPUT_FILE
echo $OUTPUT_AUX

INPUT_EDGELIST="sj_edgelist"
INPUT_NODELIST="sj_nodelist"
OUTDEGREE_BAG="tcen_at_least_five"
RECENCY_PROBABILITIES="sj_recprob"
SAME_YEAR_PROPORTION="0.12"
FULLY_RANDOM_CITATIONS="0.05"
LOG_LEVEL="1"


/usr/bin/time ./abm --edgelist ${INPUT_EDGELIST} --nodelist ${INPUT_NODELIST} --out-degree-bag ${OUTDEGREE_BAG} --recency-probabilities ${RECENCY_PROBABILITIES} --alpha 0.5 --preferential-weight 0.33 --recency-weight 0.33 --fitness-weight 0.33 --growth-rate ${GROWTH_RATE} --fully-random-citations ${FULLY_RANDOM_CITATIONS} --num-cycles ${NUM_CYCLES} --same-year-proportion ${SAME_YEAR_PROPORTION} --output-file ${OUTPUT_FILE} --auxiliary-information-file ${OUTPUT_AUX} --log-file ${OUTPUT_LOG} --num-processors ${NUM_THREADS} --log-level ${LOG_LEVEL} 2>${ERRORS}/abm.err 1>${CMDLINEOUT_FILE}
