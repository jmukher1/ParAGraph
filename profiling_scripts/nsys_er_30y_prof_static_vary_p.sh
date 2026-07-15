#!/bin/bash
#SBATCH --time=03:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --job-name="nsys-er-30y"
#SBATCH --account=ayg
#SBATCH --partition=ai
#SBATCH --mem=428GB
#SBATCH --gres=gpu:h100:1
#SBATCH --reservation=perf_count

# NO source ~/.bashrc — causes cuda/12.9 vs 12.6 version mismatch
module purge
module load modtree/gpu

ml --force unload xalt

mkdir -p output
mkdir -p errors
mkdir -p erprofile          # create profile dir alongside output/errors

OUTPUT="./output/"
ERRORS="./errors/"
PROFILE_DIR="erprofile/"   # ← define it, absolute path

NUM_THREADS=14
NUM_CYCLES=30

INPUT_EDGELIST="sj_edgelist"
INPUT_NODELIST="sj_nodelist"
OUTDEGREE_BAG="tcen_at_least_five"
RECENCY_PROBABILITIES="sj_recprob"

SAME_YEAR_PROPORTION="0.12"
FULLY_RANDOM_CITATIONS="0.05"
LOG_LEVEL="1"
#MODEL="pa"
MODEL="er-gnp"

ER_PROBABILITY="0.0001"

mkdir -p output/${NUM_CYCLES}y

growth_percents="1 3 5 6"

for GROWTH_PERCENT in $growth_percents; do
    GROWTH_RATE=$(echo "scale=2; $GROWTH_PERCENT/100" | bc)

    OUTPUT_FILE="${OUTPUT}${NUM_CYCLES}y/gpu-opt-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.edgelist"
    OUTPUT_LOG="${OUTPUT}${NUM_CYCLES}y/nsys-gpu-opt-static-${MODEL}-${NUM_CYCLES}y-${GROWTH_PERCENT}p.log"
    OUTPUT_AUX="${OUTPUT}${NUM_CYCLES}y/gpu-opt-static-output-${NUM_CYCLES}y-${GROWTH_PERCENT}p.aux"
    NSYS_OUT="${PROFILE_DIR}nsys-${MODEL}-${NUM_CYCLES}y-${GROWTH_PERCENT}p"   # ← now uses absolute PROFILE_DIR

    echo "Running growth rate ${GROWTH_PERCENT}% for ${NUM_CYCLES} years."
    echo "Profile output: ${NSYS_OUT}.nsys-rep"

    NSYS_CMD="nsys profile"
    NSYS_CMD+=" --trace=cuda,nvtx,osrt"
    NSYS_CMD+=" --gpu-metrics-device=all"
    NSYS_CMD+=" --gpu-metrics-frequency=10000"
    NSYS_CMD+=" --stats=true"
    NSYS_CMD+=" --force-overwrite=true"
    NSYS_CMD+=" -o ${NSYS_OUT}"

    APP_CMD="./abm"
    APP_CMD+=" --model ${MODEL}"
    APP_CMD+=" --er-probability ${ER_PROBABILITY}"
    APP_CMD+=" --edgelist ${INPUT_EDGELIST}"
    APP_CMD+=" --nodelist ${INPUT_NODELIST}"
    APP_CMD+=" --out-degree-bag ${OUTDEGREE_BAG}"
    APP_CMD+=" --recency-probabilities ${RECENCY_PROBABILITIES}"
    APP_CMD+=" --alpha 0.5"
    APP_CMD+=" --preferential-weight 0.33"
    APP_CMD+=" --recency-weight 0.33"
    APP_CMD+=" --fitness-weight 0.33"
    APP_CMD+=" --growth-rate ${GROWTH_RATE}"
    APP_CMD+=" --fully-random-citations ${FULLY_RANDOM_CITATIONS}"
    APP_CMD+=" --num-cycles ${NUM_CYCLES}"
    APP_CMD+=" --same-year-proportion ${SAME_YEAR_PROPORTION}"
    APP_CMD+=" --output-file ${OUTPUT_FILE}"
    APP_CMD+=" --auxiliary-information-file ${OUTPUT_AUX}"
    APP_CMD+=" --log-file ${OUTPUT_LOG}"
    APP_CMD+=" --num-processors ${NUM_THREADS}"
    APP_CMD+=" --log-level ${LOG_LEVEL}"

    echo "CMD: ${NSYS_CMD} ${APP_CMD}"
    ${NSYS_CMD} ${APP_CMD}

    # confirm immediately
    if [ -f "${NSYS_OUT}.nsys-rep" ]; then
        echo "SUCCESS: $(du -h ${NSYS_OUT}.nsys-rep | cut -f1) → ${NSYS_OUT}.nsys-rep"
    else
        echo "FAILED: no .nsys-rep at ${NSYS_OUT}.nsys-rep"
    fi

done
