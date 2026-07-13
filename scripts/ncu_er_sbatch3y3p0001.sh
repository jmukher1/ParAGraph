#!/bin/bash
#SBATCH --time=03:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --job-name="ncu-3y-3p"
#SBATCH --account=ayg
#SBATCH --partition=ai
#SBATCH --mem=128GB
#SBATCH --gres=gpu:h100:1

source /home/jmukher/.bashrc

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
MODEL="er-gnp"

ER_PROBABILITY="0.0001"
mkdir -p output/${NUM_CYCLES}y/${ER_PROBABILITY}
mkdir -p /home/jmukher/workspace/uiuc/citation_abm/profile

for GROWTH_PERCENT in $growth_percents
do
    GROWTH_RATE=$(echo "scale=2; $GROWTH_PERCENT/100" | bc)
    GROWTH_LABEL=$(echo "$GROWTH_PERCENT" | sed 's/0\./dot/; s/\./dot/')

    ER_PROBABILITY_LABEL=$(awk "BEGIN {printf \"%.4f\", $ER_PROBABILITY}")

    OUTPUT_FILE="./output/${NUM_CYCLES}y/${ER_PROBABILITY}/ncu-gpu-${MODEL}-${NUM_CYCLES}y-${GROWTH_LABEL}p-${ER_PROBABILITY_LABEL}p.edgelist"
    OUTPUT_LOG="./output/${NUM_CYCLES}y/${ER_PROBABILITY}/ncu-gpu-${MODEL}-${NUM_CYCLES}y-${GROWTH_LABEL}p-${ER_PROBABILITY_LABEL}p.log"
    OUTPUT_AUX="./output/${NUM_CYCLES}y/${ER_PROBABILITY}/ncu-gpu-${MODEL}-${NUM_CYCLES}y-${GROWTH_LABEL}p-${ER_PROBABILITY_LABEL}p.aux"
    OUT_FILE="./output/${NUM_CYCLES}y/${ER_PROBABILITY}/ncu-gpu-${MODEL}-${NUM_CYCLES}y-${GROWTH_LABEL}p-${ER_PROBABILITY_LABEL}p.out"

    echo "Running growth rate ${GROWTH_PERCENT}% for ${NUM_CYCLES} years."
    echo $OUTPUT_FILE
    echo $OUTPUT_AUX

    # Run ncu with no kernel filter and minimal set, capture exit code
    ncu \
    	--set basic \
    	--target-processes all \
   	--clock-control none \
    	--export /home/jmukher/workspace/uiuc/citation_abm/profile/debug_probe \
    	--force-overwrite \
    	./abm \
    	--model er-gnp \
    	--er-probability 0.0001 \
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
