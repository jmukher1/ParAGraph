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
#growth_percents="1 3 5 10 15"
growth_percents="3"
BFS_BATCH_SIZE=10

MODEL="er-gnp"

ER_PROBABILITY="0.0001"
ER_PROBABILITY_LABEL=$(awk "BEGIN {printf \"%.4f\", $ER_PROBABILITY}")
echo $ER_PROBABILITY_LABEL
mkdir -p output/${ER_PROBABILITY_LABEL}
for GROWTH_PERCENT in 1 3 6
do
    GROWTH_RATE=$(echo "scale=4; $GROWTH_PERCENT/100" | bc -l)
    GROWTH_LABEL=$(echo "$GROWTH_PERCENT" | sed 's/0\./dot/; s/\./dot/') 

    if [ "$GROWTH_PERCENT" -eq 5 ]; then
        BFS_BATCH_SIZE=20
        echo "For GROWTH_PERCENT = $GROWTH_PERCENT %, BFS_BATCH_SIZE = $BFS_BATCH_SIZE."
    elif [ "$GROWTH_PERCENT" -ge 6 ]; then
        BFS_BATCH_SIZE=30
        echo "For GROWTH_PERCENT = $GROWTH_PERCENT %, BFS_BATCH_SIZE = $BFS_BATCH_SIZE."
    fi

    OUTPUT_FILE="./output/${ER_PROBABILITY_LABEL}/mass-${MODEL}_cuda-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.edgelist"
    OUTPUT_LOG="./output/${ER_PROBABILITY_LABEL}/mass-${MODEL}_cuda-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.log"
    OUTPUT_AUX="./output/${ER_PROBABILITY_LABEL}/mass-${MODEL}_cuda-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.aux"
    OUT_FILE="./output/${ER_PROBABILITY_LABEL}/mass-${MODEL}_cuda-static-${NUM_CYCLES}y-${GROWTH_PERCENT}p-abm.out"

    echo "Running growth rate ${GROWTH_PERCENT}% for ${NUM_CYCLES} years."

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
ER_PROBABILITY="0.0003"
ER_PROBABILITY_LABEL=$(awk "BEGIN {printf \"%.4f\", $ER_PROBABILITY}")
mkdir -p output/${ER_PROBABILITY_LABEL}
for GROWTH_PERCENT in 1 3 6
do
    GROWTH_RATE=$(echo "scale=4; $GROWTH_PERCENT/100" | bc -l)
    GROWTH_LABEL=$(echo "$GROWTH_PERCENT" | sed 's/0\./dot/; s/\./dot/') 

    if [ "$GROWTH_PERCENT" -eq 5 ]; then
        BFS_BATCH_SIZE=20
        echo "For GROWTH_PERCENT = $GROWTH_PERCENT %, BFS_BATCH_SIZE = $BFS_BATCH_SIZE."
    elif [ "$GROWTH_PERCENT" -ge 6 ]; then
        BFS_BATCH_SIZE=30
        echo "For GROWTH_PERCENT = $GROWTH_PERCENT %, BFS_BATCH_SIZE = $BFS_BATCH_SIZE."
    fi

    OUTPUT_FILE="./output/${ER_PROBABILITY_LABEL}/mass-${MODEL}_cuda-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.edgelist"
    OUTPUT_LOG="./output/${ER_PROBABILITY_LABEL}/mass-${MODEL}_cuda-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.log"
    OUTPUT_AUX="./output/${ER_PROBABILITY_LABEL}/mass-${MODEL}_cuda-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.aux"
    OUT_FILE="./output/${ER_PROBABILITY_LABEL}/mass-${MODEL}_cuda-static-${NUM_CYCLES}y-${GROWTH_PERCENT}p-abm.out"

    echo "Running growth rate ${GROWTH_PERCENT}% for ${NUM_CYCLES} years."

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
ER_PROBABILITY="0.0005"
ER_PROBABILITY_LABEL=$(awk "BEGIN {printf \"%.4f\", $ER_PROBABILITY}")
mkdir -p output/${ER_PROBABILITY_LABEL}
for GROWTH_PERCENT in 1 3 6
do
    GROWTH_RATE=$(echo "scale=4; $GROWTH_PERCENT/100" | bc -l)
    GROWTH_LABEL=$(echo "$GROWTH_PERCENT" | sed 's/0\./dot/; s/\./dot/') 

    if [ "$GROWTH_PERCENT" -eq 5 ]; then
        BFS_BATCH_SIZE=20
        echo "For GROWTH_PERCENT = $GROWTH_PERCENT %, BFS_BATCH_SIZE = $BFS_BATCH_SIZE."
    elif [ "$GROWTH_PERCENT" -ge 6 ]; then
        BFS_BATCH_SIZE=30
        echo "For GROWTH_PERCENT = $GROWTH_PERCENT %, BFS_BATCH_SIZE = $BFS_BATCH_SIZE."
    fi

    OUTPUT_FILE="./output/${ER_PROBABILITY_LABEL}/mass-${MODEL}_cuda-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.edgelist"
    OUTPUT_LOG="./output/${ER_PROBABILITY_LABEL}/mass-${MODEL}_cuda-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.log"
    OUTPUT_AUX="./output/${ER_PROBABILITY_LABEL}/mass-${MODEL}_cuda-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.aux"
    OUT_FILE="./output/${ER_PROBABILITY_LABEL}/mass-${MODEL}_cuda-static-${NUM_CYCLES}y-${GROWTH_PERCENT}p-abm.out"

    echo "Running growth rate ${GROWTH_PERCENT}% for ${NUM_CYCLES} years."

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

ER_PROBABILITY="0.001"
ER_PROBABILITY_LABEL=$(awk "BEGIN {printf \"%.4f\", $ER_PROBABILITY}")
mkdir -p output/${ER_PROBABILITY_LABEL}
for GROWTH_PERCENT in 1 3 6
do
    GROWTH_RATE=$(echo "scale=4; $GROWTH_PERCENT/100" | bc -l)
    GROWTH_LABEL=$(echo "$GROWTH_PERCENT" | sed 's/0\./dot/; s/\./dot/') 

    if [ "$GROWTH_PERCENT" -eq 5 ]; then
        BFS_BATCH_SIZE=20
        echo "For GROWTH_PERCENT = $GROWTH_PERCENT %, BFS_BATCH_SIZE = $BFS_BATCH_SIZE."
    elif [ "$GROWTH_PERCENT" -ge 6 ]; then
        BFS_BATCH_SIZE=30
        echo "For GROWTH_PERCENT = $GROWTH_PERCENT %, BFS_BATCH_SIZE = $BFS_BATCH_SIZE."
    fi

    OUTPUT_FILE="./output/${ER_PROBABILITY_LABEL}/mass-${MODEL}_cuda-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.edgelist"
    OUTPUT_LOG="./output/${ER_PROBABILITY_LABEL}/mass-${MODEL}_cuda-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.log"
    OUTPUT_AUX="./output/${ER_PROBABILITY_LABEL}/mass-${MODEL}_cuda-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.aux"
    OUT_FILE="./output/${ER_PROBABILITY_LABEL}/mass-${MODEL}_cuda-static-${NUM_CYCLES}y-${GROWTH_PERCENT}p-abm.out"

    echo "Running growth rate ${GROWTH_PERCENT}% for ${NUM_CYCLES} years."

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
