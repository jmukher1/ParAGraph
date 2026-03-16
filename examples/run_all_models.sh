#!/bin/bash

# Example script to run all three citation models
# This script demonstrates how to compare PA, ER, and ER-GNP models

# Create output directory
mkdir -p results

echo "=========================================="
echo "Running Citation Network Growth Models"
echo "=========================================="

# Common parameters
EDGELIST="examples/seed_edgelist.txt"
NODELIST="examples/seed_nodelist.txt"
OUTDEGREE="examples/outdegree_bag.csv"
RECENCY="examples/recency_probabilities.csv"
GROWTH_RATE=0.05
NUM_CYCLES=20
SAME_YEAR=0.1
NUM_PROCS=4

echo ""
echo "1. Running Preferential Attachment Model..."
echo "   - Using degree, recency, and fitness"
./abm \
  --model pa \
  --preferential-weight 0.5 \
  --recency-weight 0.3 \
  --fitness-weight 0.2 \
  --alpha 0.7 \
  --fully-random-citations 0.05 \
  --edgelist "$EDGELIST" \
  --nodelist "$NODELIST" \
  --out-degree-bag "$OUTDEGREE" \
  --recency-probabilities "$RECENCY" \
  --growth-rate $GROWTH_RATE \
  --num-cycles $NUM_CYCLES \
  --same-year-proportion $SAME_YEAR \
  --output-file results/pa_network.txt \
  --auxiliary-information-file results/pa_attributes.txt \
  --log-file results/pa_simulation.log \
  --num-processors $NUM_PROCS \
  --log-level 1

echo ""
echo "2. Running Erdős-Rényi Fixed-k Model..."
echo "   - Uniform random citations (baseline)"
./abm \
  --model er \
  --edgelist "$EDGELIST" \
  --nodelist "$NODELIST" \
  --out-degree-bag "$OUTDEGREE" \
  --recency-probabilities "$RECENCY" \
  --growth-rate $GROWTH_RATE \
  --num-cycles $NUM_CYCLES \
  --same-year-proportion $SAME_YEAR \
  --output-file results/er_network.txt \
  --auxiliary-information-file results/er_attributes.txt \
  --log-file results/er_simulation.log \
  --num-processors $NUM_PROCS \
  --log-level 1

echo ""
echo "3. Running Erdős-Rényi G(n,p) Model..."
echo "   - Probabilistic edge formation"
./abm \
  --model er-gnp \
  --er-probability 0.01 \
  --edgelist "$EDGELIST" \
  --nodelist "$NODELIST" \
  --out-degree-bag "$OUTDEGREE" \
  --recency-probabilities "$RECENCY" \
  --growth-rate $GROWTH_RATE \
  --num-cycles $NUM_CYCLES \
  --same-year-proportion $SAME_YEAR \
  --output-file results/er_gnp_network.txt \
  --auxiliary-information-file results/er_gnp_attributes.txt \
  --log-file results/er_gnp_simulation.log \
  --num-processors $NUM_PROCS \
  --log-level 1

echo ""
echo "=========================================="
echo "All simulations completed!"
echo "=========================================="
echo ""
echo "Output files in results/:"
echo "  - *_network.txt: Edge lists"
echo "  - *_attributes.txt: Node attributes"
echo "  - *_simulation.log: Simulation logs"
echo ""
echo "You can now analyze the networks:"
echo "  python examples/analyze_networks.py"
echo ""
