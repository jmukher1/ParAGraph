# Parameter Tuning Guide

This guide helps you choose appropriate parameters for different citation network analysis scenarios.

## Model Selection

### When to Use Each Model

**Preferential Attachment (PA)** - Use when:
- You want realistic citation behavior
- Rich-get-richer dynamics are important
- Temporal effects matter (recent papers cited more)
- Papers have varying quality/fitness
- Matching real-world citation patterns

**Erdős-Rényi Fixed-k (ER)** - Use when:
- You need a null model/baseline
- Testing if observed patterns differ from random
- All citations equally likely
- Controlled out-degree distribution

**Erdős-Rényi G(n,p) (ER-GNP)** - Use when:
- Classical random graph baseline
- Variable out-degree from binomial distribution
- Theoretical comparisons needed

## Parameter Selection Guide

### 1. Growth Rate (`--growth-rate`)

**What it controls:** Fraction of new nodes added each cycle.

```
growth_rate = (new_nodes) / (existing_nodes)
```

**Typical values:**
- 0.01 (1%): Slow, mature field
- 0.05 (5%): Moderate growth (typical)
- 0.10 (10%): Rapid growth, emerging field
- 0.20 (20%): Explosive growth

**Example:**
```bash
# Slow growth (1% per year)
--growth-rate 0.01 --num-cycles 100

# Fast growth (10% per year)
--growth-rate 0.10 --num-cycles 50
```

### 2. Number of Cycles (`--num-cycles`)

**What it controls:** Number of time steps (years) to simulate.

**Considerations:**
- More cycles = larger final network
- Final size ≈ initial_size × (1 + growth_rate)^num_cycles
- Computation time grows with cycles

**Example:**
```bash
# Short simulation (20 years)
--num-cycles 20

# Long simulation (100 years)
--num-cycles 100
```

### 3. Preferential Attachment Weight (`--preferential-weight`)

**PA Model Only**

**What it controls:** How much degree influences citation probability.

**Values:**
- 0.0: No preferential attachment (uniform)
- 0.5: Moderate preference for popular papers
- 1.0: Strong rich-get-richer effect
- -1: Random weight (sampled uniformly)

**Effect on network:**
- Higher → More concentrated citations, higher max degree
- Lower → More uniform distribution

**Example:**
```bash
# Strong preferential attachment
--preferential-weight 0.8 --recency-weight 0.1 --fitness-weight 0.1

# Balanced model
--preferential-weight 0.33 --recency-weight 0.33 --fitness-weight 0.34

# Fitness-driven
--preferential-weight 0.1 --recency-weight 0.1 --fitness-weight 0.8
```

### 4. Recency Weight (`--recency-weight`)

**PA Model Only**

**What it controls:** How much publication recency influences citation.

**Values:**
- 0.0: No recency bias (all years equal)
- 0.5: Moderate preference for recent papers
- 1.0: Strong bias toward recent papers
- -1: Random weight

**Effect on network:**
- Higher → Recent papers get more citations
- Lower → Historical papers remain competitive

**Example:**
```bash
# Strong recency bias (citation amnesia)
--recency-weight 0.7

# Weak recency bias (long memory)
--recency-weight 0.2
```

### 5. Fitness Weight (`--fitness-weight`)

**PA Model Only**

**What it controls:** How much intrinsic quality influences citation.

**Values:**
- 0.0: Quality doesn't matter
- 0.5: Quality moderately important
- 1.0: Quality dominates
- -1: Random weight

**Effect on network:**
- Higher → High-fitness papers accumulate citations
- Lower → All papers more equal

**Example:**
```bash
# Quality-driven citations
--fitness-weight 0.6 --preferential-weight 0.2 --recency-weight 0.2
```

### 6. Alpha Parameter (`--alpha`)

**PA Model Only**

**What it controls:** Proportion of citations from 1-hop vs 2-hop neighborhood.

**Values:**
- 0.0: All citations from 2-hop neighborhood
- 0.5: Equal split between 1-hop and 2-hop
- 1.0: All citations from 1-hop neighborhood
- -1: Random alpha

**Effect on network:**
- Higher → More local clustering
- Lower → More exploration, broader reach

**Example:**
```bash
# High clustering (local citations)
--alpha 0.9

# Broad exploration
--alpha 0.3
```

### 7. Fully Random Citations (`--fully-random-citations`)

**PA Model Only**

**What it controls:** Fraction of completely random citations (exploration).

**Values:**
- 0.0: No random citations
- 0.05: 5% random (typical)
- 0.10: 10% random (high exploration)

**Effect on network:**
- Higher → More serendipitous discoveries
- Lower → More constrained to neighborhood

**Example:**
```bash
# Low exploration
--fully-random-citations 0.02

# High exploration
--fully-random-citations 0.15
```

### 8. Edge Probability (`--er-probability`)

**ER-GNP Model Only**

**What it controls:** Probability of creating each edge.

**Typical values:**
- 0.001 (0.1%): Sparse network
- 0.01 (1%): Moderate density
- 0.05 (5%): Dense network

**Note:** Expected degree = p × n (where n is graph size)

**Example:**
```bash
# Sparse ER network (avg degree ≈ 10 for 1000 nodes)
--er-probability 0.01 --num-cycles 50

# Dense ER network
--er-probability 0.05 --num-cycles 20
```

### 9. Same-Year Proportion (`--same-year-proportion`)

**What it controls:** Fraction of papers citing same-year publications.

**Values:**
- 0.0: No same-year citations
- 0.1: 10% same-year (typical)
- 0.3: 30% same-year (high)

**Example:**
```bash
# Low same-year citations
--same-year-proportion 0.05

# High same-year citations (fast-paced field)
--same-year-proportion 0.25
```

### 10. Number of Processors (`--num-processors`)

**What it controls:** OpenMP thread count for parallelization.

**Values:**
- 1: No parallelization
- 4-8: Moderate parallelization
- 16+: High parallelization (for large networks)

**Recommendation:** Set to number of CPU cores available.

```bash
# Check available cores
nproc

# Use 8 cores
--num-processors 8
```

## Preset Configurations

### Classic Barabási-Albert Style

```bash
./citation_models --model pa \
  --preferential-weight 1.0 \
  --recency-weight 0.0 \
  --fitness-weight 0.0 \
  --alpha 0.5 \
  --fully-random-citations 0.0
```

### Realistic Citation Network

```bash
./citation_models --model pa \
  --preferential-weight 0.4 \
  --recency-weight 0.4 \
  --fitness-weight 0.2 \
  --alpha 0.7 \
  --fully-random-citations 0.05
```

### Fitness-Dominated Network

```bash
./citation_models --model pa \
  --preferential-weight 0.2 \
  --recency-weight 0.1 \
  --fitness-weight 0.7 \
  --alpha 0.6 \
  --fully-random-citations 0.05
```

### Random Baseline (ER Fixed-k)

```bash
./citation_models --model er \
  --growth-rate 0.05 \
  --num-cycles 50
```

### Sparse Random Graph (ER-GNP)

```bash
./citation_models --model er-gnp \
  --er-probability 0.005 \
  --growth-rate 0.05 \
  --num-cycles 50
```

## Parameter Tuning Workflow

### 1. Start with Example Parameters

Run with default/example parameters to understand baseline behavior.

### 2. Match Real Data Statistics

If you have real citation data:

```python
# Compute target statistics
import networkx as nx

G_real = nx.read_edgelist('real_citations.txt', create_using=nx.DiGraph())
target_avg_degree = sum(dict(G_real.in_degree()).values()) / G_real.number_of_nodes()
target_max_degree = max(dict(G_real.in_degree()).values())
```

### 3. Iterate on Parameters

```bash
# Try different PA weights
for pa_w in 0.3 0.5 0.7; do
  ./citation_models --model pa --preferential-weight $pa_w ...
done

# Try different ER probabilities
for p in 0.005 0.01 0.02; do
  ./citation_models --model er-gnp --er-probability $p ...
done
```

### 4. Evaluate Fit

```python
import pandas as pd
import networkx as nx

# Load simulated network
G_sim = nx.read_edgelist('simulation.txt', create_using=nx.DiGraph())

# Compare degree distributions
from scipy.stats import ks_2samp

real_degrees = [d for n, d in G_real.in_degree()]
sim_degrees = [d for n, d in G_sim.in_degree()]

# Kolmogorov-Smirnov test
statistic, pvalue = ks_2samp(real_degrees, sim_degrees)
print(f"KS statistic: {statistic}, p-value: {pvalue}")
```

### 5. Refine

Adjust parameters based on which statistics don't match:
- Too high max degree → Lower preferential weight
- Too uniform → Raise preferential weight
- Wrong temporal pattern → Adjust recency weight
- Wrong clustering → Adjust alpha

## Common Scenarios

### Scenario 1: Matching Real Citation Network

**Goal:** Replicate properties of an observed citation network.

**Steps:**
1. Compute statistics from real network (degree dist, clustering, etc.)
2. Start with PA model, balanced weights
3. Adjust preferential weight to match degree heterogeneity
4. Adjust recency weight to match temporal patterns
5. Tune alpha to match clustering coefficient

### Scenario 2: Null Model Testing

**Goal:** Test if observed patterns differ from random.

**Steps:**
1. Use ER model with same out-degree distribution
2. Compare real vs. ER on key metrics
3. Statistical test (e.g., permutation test)

### Scenario 3: Mechanism Investigation

**Goal:** Understand which mechanisms drive observed patterns.

**Steps:**
1. Run PA with all weights (full model)
2. Run PA with only preferential attachment
3. Run PA with only recency
4. Run PA with only fitness
5. Compare to identify dominant mechanism

## Performance Considerations

### Memory Usage

Approximate memory per node:
- PA model: ~500 bytes/node
- ER models: ~200 bytes/node

For 1M nodes:
- PA: ~500 MB
- ER: ~200 MB

### Computation Time

Rough estimates (on modern CPU, 8 cores):
- 1000 nodes, 20 cycles: <1 minute
- 10,000 nodes, 50 cycles: 5-15 minutes
- 100,000 nodes, 100 cycles: 1-3 hours
- 1,000,000 nodes: Hours to days

**Speed tips:**
- Use ER models for quick tests (faster than PA)
- Reduce --num-cycles for prototyping
- Increase --num-processors
- Set --log-level 0 (logging is expensive)

## Validation Checklist

- [ ] Final network size makes sense
- [ ] Degree distribution looks reasonable
- [ ] Temporal growth pattern is correct
- [ ] No disconnected components (unless expected)
- [ ] Log file shows no errors
- [ ] Output files are complete

## Further Reading

- Barabási & Albert (1999): "Emergence of scaling in random networks"
- Erdős & Rényi (1959): "On random graphs"
- Wang et al. (2013): "Quantifying long-term scientific impact"
- Your citation network papers here!
