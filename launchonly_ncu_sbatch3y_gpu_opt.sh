#!/bin/bash
#SBATCH --time=00:03:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --job-name="3y-3p-lncu"
#SBATCH --account=zgdrasil
#SBATCH --partition=zgdrasil
#SBATCH --mem=64GB
#SBATCH --gres=gpu:A30

source ~/.bashrc

# Load necessary modules (e.g., CUDA)
module load cuda/12.6

# Vary growth rate from 1% to 10%
# 3 years simulation
# 1 thread

mkdir -p output
mkdir -p errors

OUTPUT=$(readlink -f ./output)/
ERRORS=$(readlink -f ./errors)/

NUM_THREADS=16
NUM_CYCLES=3

INPUT_EDGELIST="sj_edgelist"
INPUT_NODELIST="sj_nodelist"
OUTDEGREE_BAG="tcen_at_least_five"
RECENCY_PROBABILITIES="sj_recprob"

SAME_YEAR_PROPORTION="0.12"
FULLY_RANDOM_CITATIONS="0.05"
LOG_LEVEL="1"

#growth_percents="1 3 5 7 9 11 15 20 25 50 75 100"
growth_percents="3"
MODEL="pa"

mkdir -p output/${NUM_CYCLES}y/${ER_PROBABILITY}
mkdir -p profile

for GROWTH_PERCENT in $growth_percents
do
    GROWTH_RATE=$(echo "scale=2; $GROWTH_PERCENT/100" | bc)
    GROWTH_LABEL=$(echo "$GROWTH_PERCENT" | sed 's/0\./dot/; s/\./dot/')


    OUTPUT_FILE="./output/${NUM_CYCLES}y/ncu-gpu-${MODEL}-${NUM_CYCLES}y-${GROWTH_LABEL}p.edgelist"
    OUTPUT_LOG="./output/${NUM_CYCLES}y/ncu-gpu-${MODEL}-${NUM_CYCLES}y-${GROWTH_LABEL}p.log"
    OUTPUT_AUX="./output/${NUM_CYCLES}y/ncu-gpu-${MODEL}-${NUM_CYCLES}y-${GROWTH_LABEL}p.aux"
    OUT_FILE="./output/${NUM_CYCLES}y/ncu-gpu-${MODEL}-${NUM_CYCLES}y-${GROWTH_LABEL}p.out"

    echo "Running growth rate ${GROWTH_PERCENT}% for ${NUM_CYCLES} years."
    echo $OUTPUT_FILE
    echo $OUTPUT_AUX

    # Run ncu with no kernel filter and minimal set, capture exit code
    ncu \
	--metrics \
        launch__registers_per_thread,\
        launch__shared_mem_per_block_static,\
        launch__shared_mem_per_block_dynamic,\
        launch__occupancy_theoretical,\
        launch__waves_per_multiprocessor,\
        launch__block_size,\
        launch__grid_size,\
        launch__thread_count \
    	--clock-control none \
    	--csv \
    	--export "profile/ncu_launch_${MODEL}_${GROWTH_LABEL}p" \
    	--force-overwrite \
        ./abm \
        --edgelist "${INPUT_EDGELIST}" \
        --nodelist "${INPUT_NODELIST}" \
        --out-degree-bag "${OUTDEGREE_BAG}" \
        --recency-probabilities "${RECENCY_PROBABILITIES}" \
        --alpha 0.5 \
        --preferential-weight 0.33 \
        --recency-weight 0.33 \
        --fitness-weight 0.33 \
        --growth-rate 0.03 \
        --fully-random-citations 0.05 \
        --num-cycles 3 \
        --same-year-proportion 0.12 \
        --output-file "${OUTPUT_FILE}" \
        --auxiliary-information-file "${OUTPUT_AUX}" \
        --log-file "${OUTPUT_LOG}" \
        --num-processors "${NUM_THREADS}" \
        --log-level 1

    echo "ncu exit code: $?"
    echo "---"
    grep -E "PROF|WARN|ERROR|kernel|Kernel" /tmp/ncu_debug_full.log

done
