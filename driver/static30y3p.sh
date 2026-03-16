#!/bin/sh
# 30 years 3% growth - static

#./abm --edgelist sj_edgelist --nodelist sj_nodelist --out-degree-bag tcen_at_least_five --recency-probabilities sj_recprob --same-year-proportion 0.12 
#--growth-rate 0.03 --auxiliary-information-file cpu-dummy30-3.aux --num-cycles 30 --output-file cpu-res30-3.out --log-file cpu-output30-3.log --num-processors 64 --log-level 1


mkdir -p output
mkdir -p errors
OUTPUT=$(readlink -f ./output)/
ERRORS=$(readlink -f ./errors)/

NUM_THREADS=64
GROWTH_RATE=0.03
NUM_CYCLES=30

GROWTH_PERCENT=$(echo "$GROWTH_RATE * 100" | bc | cut -d. -f1)
OUTPUT_FILE="./output/cpu-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.edgelist"
OUTPUT_LOG="./output/cpu-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.log"
OUTPUT_AUX="./output/cpu-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.aux"
OUT_FILE="./output/cpu-static-${NUM_CYCLES}y-${GROWTH_PERCENT}p-abm.out"

echo $OUTPUT_FILE
echo $OUTPUT_AUX

INPUT_EDGELIST="sj_edgelist"
INPUT_NODELIST="sj_nodelist"
OUTDEGREE_BAG="tcen_at_least_five"
RECENCY_PROBABILITIES="sj_recprob"
SAME_YEAR_PROPORTION="0.12"
FULLY_RANDOM_CITATIONS="0.05"
LOG_LEVEL="1"


/usr/bin/time -v ./abm --edgelist ${INPUT_EDGELIST} --nodelist ${INPUT_NODELIST} --out-degree-bag ${OUTDEGREE_BAG} --recency-probabilities ${RECENCY_PROBABILITIES} --alpha 0.5 --preferential-weight 0.33 --recency-weight 0.33 --fitness-weight 0.33 --growth-rate ${GROWTH_RATE} --fully-random-citations ${FULLY_RANDOM_CITATIONS} --num-cycles ${NUM_CYCLES} --same-year-proportion ${SAME_YEAR_PROPORTION} --output-file ${OUTPUT_FILE} --auxiliary-information-file ${OUTPUT_AUX} --log-file ${OUTPUT_LOG} --num-processors ${NUM_THREADS} --log-level ${LOG_LEVEL} 2>${ERRORS}/abm.err 1>${OUT_FILE}
