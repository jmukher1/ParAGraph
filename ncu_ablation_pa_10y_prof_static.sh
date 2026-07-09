#!/bin/bash
#SBATCH --time=09:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --job-name="ncu-pa-10y"
#SBATCH --account=ayg
#SBATCH --partition=ai
#SBATCH --gres=gpu:h100:1

# NO source ~/.bashrc — causes cuda/12.9 vs 12.6 version mismatch
module purge
module load modtree/gpu

ml --force unload xalt

module list

echo "=== ENV CHECK ==="
echo "nsys:  $(which nsys)  |  $(nsys --version 2>&1 | head -1)"
echo "ncu:   $(which ncu)   |  $(ncu --version  2>&1 | head -1)"
echo "nvcc:  $(nvcc --version 2>&1 | grep release)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=================="

ABM_BIN="./abm"
PROFILE_DIR="./profile"
ERRORS="./errors"

NUM_THREADS=${SLURM_CPUS_PER_TASK}
NUM_CYCLES=10
mkdir -p "${PROFILE_DIR}" "${ERRORS}" 

# All input files as absolute paths
INPUT_EDGELIST="sj_edgelist"
INPUT_NODELIST="sj_nodelist"
OUTDEGREE_BAG="tcen_at_least_five"
RECENCY_PROBS="sj_recprob"

MODEL="pa"

mkdir -p output
mkdir -p profile

SAME_YEAR_PROPORTION="0.12"
FULLY_RANDOM_CITATIONS="0.05"
LOG_LEVEL="1"

# ── Ablation configurations ──────────────────────────────────────────────────
# label|use_warp_bfs|use_batching|max_batch_size|use_multistage_kernel
CONFIGS=(
    "baseline|true|true|20000|true"
    "no-warp|false|true|20000|true"
    "no-batching|true|false|20000|true"
    "large-batch|true|true|1000000|true"
    "monolithic|true|true|20000|false"
)

CONFIGS=(
    "baseline|true|true|100000|true"
    "default-batch|true|true|20000|true"
    "no-warp|false|true|20000|true"
    "no-batching|true|false|20000|true"
    "large-batch|true|true|1000000|true"
    "monolithic|true|true|20000|false"
)

# ── metrics: roofline + occupancy + memory bandwidth ─────────────────
# These require hardware counters — only works with --reservation=perf_count
NCU_METRICS=""
# Occupancy
NCU_METRICS+="sm__warps_active.avg.pct_of_peak_sustained_active,"
# DRAM bandwidth utilization
NCU_METRICS+="gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,"
# FP32 and FP64 FLOP counts (for roofline arithmetic intensity)
NCU_METRICS+="sm__sass_thread_inst_executed_op_fp32_pred_on.sum,"
NCU_METRICS+="sm__sass_thread_inst_executed_op_fp64_pred_on.sum,"
NCU_METRICS+="lts__t_sector_hit_rate.pct,"
NCU_METRICS+="smsp__thread_inst_executed_per_inst_executed.ratio,"
# Total DRAM bytes (for roofline denominator)
NCU_METRICS+="dram__bytes.sum,"
# Kernel wall time
NCU_METRICS+="gpu__time_duration.sum,"
# L2 cache hit rate
NCU_METRICS+="lts__t_sectors_srcunit_tex_op_read_hit_rate.pct,"
# SM compute throughput
NCU_METRICS+="sm__throughput.avg.pct_of_peak_sustained_elapsed,"
# Launch statistics (no counter needed, but included for completeness)
NCU_METRICS+="launch__registers_per_thread,"
NCU_METRICS+="launch__shared_mem_per_block_static,"
NCU_METRICS+="launch__shared_mem_per_block_dynamic,"
NCU_METRICS+="launch__occupancy_theoretical,"
NCU_METRICS+="launch__waves_per_multiprocessor,"
NCU_METRICS+="launch__block_size,"
NCU_METRICS+="launch__grid_size,"
NCU_METRICS+="launch__thread_count"
# strip trailing comma if any
NCU_METRICS="${NCU_METRICS%,}"

# ── kernel-name filter, broadened to cover every ablation config ────────────
# baseline/large-batch/no-batching (multistage+warp)   -> kernelCallStage1_warped, kernelCallStage2/3/4
# no-warp                          (multistage+no-warp) -> kernelCallStage1 (non-warped), kernelCallStage2/3/4
# monolithic                       (fused)               -> ABMKernel only (Stage1-4 never launch separately)
# A single regex covering all of these avoids needing a per-config branch,
# and avoids relying on whether ncu's --kernel-name does substring vs.
# full-string regex matching (kernelCallStage[1-4] alone would ambiguously
# match or fail to match "kernelCallStage1_warped" depending on that,
# and would match nothing at all for the monolithic config).
NCU_KERNEL_REGEX="kernelCallStage[1-4]|kernelCallStage1_warped|ABMKernel"

growth_percents="1 3 6"

for GROWTH_PERCENT in $growth_percents; do
    GROWTH_RATE=$(echo "scale=2; $GROWTH_PERCENT/100" | bc)

    for entry in "${CONFIGS[@]}"; do
        IFS='|' read -r LABEL USE_WARP_BFS USE_BATCHING MAX_BATCH_SIZE USE_MULTISTAGE_KERNEL <<< "$entry"

        mkdir -p "./output/ablation/ncu-H100"
        mkdir -p "${PROFILE_DIR}/${LABEL}"

        OUTPUT_BASE="./output/ablation/ncu-H100/${LABEL}-ncu-gpu-parallel-static-${NUM_CYCLES}y-${GROWTH_PERCENT}p"
        PROFILE_BASE="${PROFILE_DIR}/${LABEL}-ncu-gpu-parallel-static-${MODEL}-${NUM_CYCLES}y-${GROWTH_PERCENT}p"

        echo ""
        echo "=== Ablation config: ${LABEL}  |  Growth ${GROWTH_PERCENT}% / ${NUM_CYCLES}y ==="
        echo "    use-warp-bfs=${USE_WARP_BFS}  use-batching=${USE_BATCHING}  max-batch-size=${MAX_BATCH_SIZE}  use-multistage-kernel=${USE_MULTISTAGE_KERNEL}"
        echo "    Profile: ${PROFILE_BASE}.ncu-rep"
        echo "    CSV:     ${PROFILE_BASE}.csv"
        echo "    ${OUTPUT_BASE}.edgelist"
        echo "    ${OUTPUT_BASE}.aux"

        APP_CMD="./abm"
        APP_CMD+=" --edgelist ${INPUT_EDGELIST}"
        APP_CMD+=" --nodelist ${INPUT_NODELIST}"
        APP_CMD+=" --out-degree-bag ${OUTDEGREE_BAG}"
        APP_CMD+=" --recency-probabilities ${RECENCY_PROBS}"
        APP_CMD+=" --alpha 0.5"
        APP_CMD+=" --preferential-weight 0.33"
        APP_CMD+=" --recency-weight 0.33"
        APP_CMD+=" --fitness-weight 0.33"
        APP_CMD+=" --growth-rate ${GROWTH_RATE}"
        APP_CMD+=" --fully-random-citations ${FULLY_RANDOM_CITATIONS}"
        APP_CMD+=" --num-cycles ${NUM_CYCLES}"
        APP_CMD+=" --same-year-proportion ${SAME_YEAR_PROPORTION}"
        APP_CMD+=" --output-file ${OUTPUT_BASE}.edgelist"
        APP_CMD+=" --auxiliary-information-file ${OUTPUT_BASE}.aux"
        APP_CMD+=" --log-file ${OUTPUT_BASE}.log"
        APP_CMD+=" --num-processors ${NUM_THREADS}"
        APP_CMD+=" --log-level ${LOG_LEVEL}"
        APP_CMD+=" --use-warp-bfs ${USE_WARP_BFS}"
        APP_CMD+=" --use-batching ${USE_BATCHING}"
        APP_CMD+=" --max-batch-size ${MAX_BATCH_SIZE}"
        APP_CMD+=" --use-multistage-kernel ${USE_MULTISTAGE_KERNEL}"

        ncu \
            --metrics "${NCU_METRICS}" \
            --kernel-name regex:"${NCU_KERNEL_REGEX}" \
            --target-processes all \
            --clock-control none \
            --export "${PROFILE_BASE}" \
            --force-overwrite \
            --csv \
            ${APP_CMD} \
            > "${PROFILE_BASE}.csv" \
            2>"${ERRORS}/ncu-${LABEL}-${MODEL}-${NUM_CYCLES}y-${GROWTH_PERCENT}p.err" \
            1>"${OUTPUT_BASE}.out"

        NCU_EXIT=$?
        echo "    ncu exit code: ${NCU_EXIT}"

        if [ -f "${PROFILE_BASE}.ncu-rep" ]; then
            echo "    SUCCESS .ncu-rep: $(du -h ${PROFILE_BASE}.ncu-rep | cut -f1)"
            echo "    SUCCESS .csv:     $(wc -l < ${PROFILE_BASE}.csv) lines"
        else
            echo "    FAILED — no .ncu-rep produced"
            echo "    --- stderr (first 30 lines) ---"
            head -30 "${ERRORS}/ncu-${LABEL}-${MODEL}-${NUM_CYCLES}y-${GROWTH_PERCENT}p.err"
        fi

    done
done

echo ""
echo "All ablation runs complete. Compare wall-clock time / epoch breakdown across:"
for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r LABEL _ _ _ _ <<< "$entry"
    echo "  ${LABEL}"
done

echo ""
echo "=== ALL PROFILES ==="
find "${PROFILE_DIR}" -name "*.ncu-rep" -exec ls -lh {} \; 2>/dev/null || echo "no .ncu-rep files"
find "${PROFILE_DIR}" -name "*.csv"     -exec ls -lh {} \; 2>/dev/null || echo "no .csv files"