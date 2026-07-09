#!/bin/bash
# Ablation study: sj dataset, 10 years, 6% growth (fixed scenario)
#
# ASSUMPTION TO VERIFY BEFORE THE FULL RUN: this script assumes the CLI
# accepts explicit boolean values, e.g. `--use-warp-bfs false`, rather than
# an implicit presence-only flag style (e.g. a separate `--no-use-warp-bfs`
# negation flag). Confirm with:
#     ./abm --help
# and adjust the CONFIGS array / invocation below if the real syntax differs.
#
# Ablation parameters (per ABM class):
#   --use-warp-bfs {true,false}          default: true  -- warp-cooperative
#       vs per-thread BFS for Stage 1 (kernelCallStage1_warped vs
#       kernelCallStage1)
#   --use-batching {true,false}          default: true  -- adaptive batching
#       vs single batch covering all of a year's new nodes
#   --max-batch-size <N>                 default: 20000 -- only takes effect
#       when --use-batching true; always passed explicitly below regardless
#       of --use-batching's value, in case the flag is non-optional
#   --use-multistage-kernel {true,false} default: true  -- 4 separate kernel
#       launches (kernelCallStage1..4) vs one fused ABMKernel launch
#
# CONFIGS below (label|use_warp_bfs|use_batching|max_batch_size|use_multistage_kernel):
#   baseline     : all defaults (warp BFS, adaptive batching @20000, multistage)
#   no-warp      : Stage 1 divergence/occupancy ablation -- per-thread BFS instead of warp-cooperative
#   no-batching  : batching ablation -- single batch per year instead of adaptive batching
#   large-batch  : batching ablation -- adaptive batching with a much larger ceiling (1M).
#                  effectively behaves like no-batching for this dataset size
#                  (max_batch_size will almost never bind), included to
#                  isolate whether large-but-still-adaptive batching differs
#                  from truly disabling batching altogether (no-batching)
#   monolithic   : scheduling ablation -- single fused ABMKernel instead of
#                  4 separate kernel launches per batch
#
# "baseline" is not one of the 4 configs explicitly requested, but is
# included so the other four have something to compare against -- an
# ablation delta is only meaningful relative to a reference run. Comment out
# the "baseline" line below if you specifically want just the 4.

mkdir -p output/ablation
mkdir -p errors

OUTPUT=$(readlink -f ./output)/
ERRORS=$(readlink -f ./errors)/

NUM_THREADS=16
NUM_CYCLES=10 

INPUT_EDGELIST="sj_edgelist"
INPUT_NODELIST="sj_nodelist"
OUTDEGREE_BAG="tcen_at_least_five"
RECENCY_PROBABILITIES="sj_recprob"

SAME_YEAR_PROPORTION="0.12"
FULLY_RANDOM_CITATIONS="0.05"
LOG_LEVEL="1"

# ── Ablation configurations ──────────────────────────────────────────────────
CONFIGS=(
    "baseline|true|true|100000|true"
    "default-batch|true|true|20000|true"
    "no-warp|false|true|20000|true"
    "no-batching|true|false|20000|true"
    "large-batch|true|true|1000000|true"
    "monolithic|true|true|20000|false"
)

growth_percents="1 3 6"

for GROWTH_PERCENT in $growth_percents; do
    GROWTH_RATE=$(echo "scale=2; $GROWTH_PERCENT/100" | bc)
    for entry in "${CONFIGS[@]}"
    do
        IFS='|' read -r LABEL USE_WARP_BFS USE_BATCHING MAX_BATCH_SIZE USE_MULTISTAGE_KERNEL <<< "$entry"

        OUTPUT_FILE="./output/ablation/${LABEL}-gpu-parallel-static-${NUM_CYCLES}y-${GROWTH_PERCENT}p.edgelist"
        OUTPUT_LOG="./output/ablation/${LABEL}-gpu-parallel-static-${NUM_CYCLES}y-${GROWTH_PERCENT}p-${NUM_THREADS}t.log"
        OUTPUT_AUX="./output/ablation/${LABEL}-gpu-parallel-static-${NUM_CYCLES}y-${GROWTH_PERCENT}p.aux"
        OUT_FILE="./output/ablation/${LABEL}-gpu-parallel-static-${NUM_CYCLES}y-${GROWTH_PERCENT}p-${NUM_THREADS}t.out"

        echo "=== Ablation config: ${LABEL} ==="
        echo "    use-warp-bfs=${USE_WARP_BFS}  use-batching=${USE_BATCHING}  max-batch-size=${MAX_BATCH_SIZE}  use-multistage-kernel=${USE_MULTISTAGE_KERNEL}"
        echo "    ${OUTPUT_FILE}"
        echo "    ${OUTPUT_AUX}"

        ./abm \
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
        --log-file ${OUTPUT_LOG} \
        --num-processors ${NUM_THREADS} \
        --log-level ${LOG_LEVEL} \
        --use-warp-bfs ${USE_WARP_BFS} \
        --use-batching ${USE_BATCHING} \
        --max-batch-size ${MAX_BATCH_SIZE} \
        --use-multistage-kernel ${USE_MULTISTAGE_KERNEL} \
        2>${ERRORS}/abm-${LABEL}-${NUM_CYCLES}y-${GROWTH_PERCENT}p.err \
        1>${OUT_FILE}

    done
done

echo ""
echo "All ablation runs complete. Compare wall-clock time / epoch breakdown across:"
for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r LABEL _ _ _ _ <<< "$entry"
    echo "  ${LABEL}"
done
