#!/bin/bash

# Comprehensive test script for citation_models
# Tests all three models with minimal parameters

echo "=========================================="
echo "Citation Models - Comprehensive Test"
echo "=========================================="
echo ""

# Check if executable exists
if [ ! -f "./citation_models" ]; then
    echo "Error: citation_models executable not found!"
    echo "Please run 'make' first."
    exit 1
fi

# Create results directory
mkdir -p test_results

# Test parameters (small for quick test)
EDGELIST="examples/seed_edgelist.txt"
NODELIST="examples/seed_nodelist.txt"
OUTDEGREE="examples/outdegree_bag.csv"
RECENCY="examples/recency_probabilities.csv"
GROWTH=0.10
CYCLES=5
SAME_YEAR=0.1
PROCS=2

echo "Test Configuration:"
echo "  Growth rate: $GROWTH"
echo "  Num cycles: $CYCLES"
echo "  Processors: $PROCS"
echo ""

# Test 1: PA Model
echo "Test 1/3: Preferential Attachment Model"
echo "  Running..."
./citation_models \
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
  --growth-rate $GROWTH \
  --num-cycles $CYCLES \
  --same-year-proportion $SAME_YEAR \
  --output-file test_results/test_pa_network.txt \
  --auxiliary-information-file test_results/test_pa_attrs.txt \
  --log-file test_results/test_pa.log \
  --num-processors $PROCS \
  --log-level 1 > /dev/null 2>&1

if [ $? -eq 0 ]; then
    EDGES=$(wc -l < test_results/test_pa_network.txt | tr -d ' ')
    echo "  ✓ PA model completed successfully"
    echo "    Output: $EDGES edges generated"
else
    echo "  ✗ PA model failed"
    exit 1
fi
echo ""

# Test 2: ER Fixed-k Model
echo "Test 2/3: Erdős-Rényi Fixed-k Model"
echo "  Running..."
./citation_models \
  --model er \
  --edgelist "$EDGELIST" \
  --nodelist "$NODELIST" \
  --out-degree-bag "$OUTDEGREE" \
  --recency-probabilities "$RECENCY" \
  --growth-rate $GROWTH \
  --num-cycles $CYCLES \
  --same-year-proportion $SAME_YEAR \
  --output-file test_results/test_er_network.txt \
  --auxiliary-information-file test_results/test_er_attrs.txt \
  --log-file test_results/test_er.log \
  --num-processors $PROCS \
  --log-level 1 > /dev/null 2>&1

if [ $? -eq 0 ]; then
    EDGES=$(wc -l < test_results/test_er_network.txt | tr -d ' ')
    echo "  ✓ ER fixed-k model completed successfully"
    echo "    Output: $EDGES edges generated"
else
    echo "  ✗ ER fixed-k model failed"
    exit 1
fi
echo ""

# Test 3: ER G(n,p) Model
echo "Test 3/3: Erdős-Rényi G(n,p) Model"
echo "  Running..."
./citation_models \
  --model er-gnp \
  --er-probability 0.05 \
  --edgelist "$EDGELIST" \
  --nodelist "$NODELIST" \
  --out-degree-bag "$OUTDEGREE" \
  --recency-probabilities "$RECENCY" \
  --growth-rate $GROWTH \
  --num-cycles $CYCLES \
  --same-year-proportion $SAME_YEAR \
  --output-file test_results/test_er_gnp_network.txt \
  --auxiliary-information-file test_results/test_er_gnp_attrs.txt \
  --log-file test_results/test_er_gnp.log \
  --num-processors $PROCS \
  --log-level 1 > /dev/null 2>&1

if [ $? -eq 0 ]; then
    EDGES=$(wc -l < test_results/test_er_gnp_network.txt | tr -d ' ')
    echo "  ✓ ER G(n,p) model completed successfully"
    echo "    Output: $EDGES edges generated"
else
    echo "  ✗ ER G(n,p) model failed"
    exit 1
fi
echo ""

# Summary
echo "=========================================="
echo "All Tests Passed Successfully!"
echo "=========================================="
echo ""
echo "Test outputs in test_results/:"
ls -lh test_results/
echo ""
echo "You can now run the full examples:"
echo "  ./examples/run_all_models.sh"
echo ""
