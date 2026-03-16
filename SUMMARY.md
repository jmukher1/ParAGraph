# Citation Network Growth Models - Project Summary

## Overview

This package implements three citation network growth models in C++ with OpenMP parallelization:

1. **Preferential Attachment (PA)** - Realistic citation dynamics with degree, recency, and fitness
2. **Erdős-Rényi Fixed-k (ER)** - Uniform random baseline with controlled out-degree
3. **Erdős-Rényi G(n,p) (ER-GNP)** - Classical random graph with probability p

## Quick Navigation

### Getting Started
- **[QUICKSTART.md](QUICKSTART.md)** - Build and run in 5 minutes
- **[README.md](README.md)** - Complete documentation
- **[PARAMETERS.md](PARAMETERS.md)** - Parameter tuning guide

### Running Simulations
```bash
# 1. Build
make

# 2. Test
./test.sh

# 3. Run examples
./examples/run_all_models.sh

# 4. Analyze
python3 examples/analyze_networks.py
```

## Project Structure

```
citation_models/
│
├── README.md                    # Full documentation
├── QUICKSTART.md                # Quick start guide
├── PARAMETERS.md                # Parameter tuning guide
├── makefile                     # Build configuration
├── test.sh                      # Quick test script
│
├── src/                         # Source code
│   ├── main.cpp                # Entry point & CLI
│   ├── abm.cpp                 # Core simulation logic
│   └── graph.cpp               # Graph data structure
│
├── includes/                    # Header files
│   ├── abm.h                   # ABM class
│   ├── graph.h                 # Graph class
│   └── argparse.h              # CLI parsing
│
├── examples/                    # Example data & scripts
│   ├── seed_edgelist.txt       # Example seed graph
│   ├── seed_nodelist.txt       # Example node years
│   ├── outdegree_bag.csv       # Example degree dist
│   ├── recency_probabilities.csv # Example recency
│   ├── run_all_models.sh       # Run all 3 models
│   └── analyze_networks.py     # Python analysis
│
└── build/                       # Compiled objects (created)
```

## Models at a Glance

### Preferential Attachment (--model pa)
**Best for:** Realistic citation networks

**Key features:**
- Rich-get-richer dynamics
- Temporal recency effects
- Intrinsic fitness/quality
- Network neighborhood structure

**Command:**
```bash
./citation_models --model pa \
  --preferential-weight 0.5 \
  --recency-weight 0.3 \
  --fitness-weight 0.2 \
  --alpha 0.7 \
  [common parameters...]
```

### Erdős-Rényi Fixed-k (--model er)
**Best for:** Null model / baseline comparison

**Key features:**
- Uniform random target selection
- Controlled out-degree from bag
- No preferential dynamics

**Command:**
```bash
./citation_models --model er \
  [common parameters...]
```

### Erdős-Rényi G(n,p) (--model er-gnp)
**Best for:** Classical random graph comparison

**Key features:**
- Independent edge probability p
- Variable out-degree (binomial)
- Theoretical baseline

**Command:**
```bash
./citation_models --model er-gnp \
  --er-probability 0.01 \
  [common parameters...]
```

## Common Parameters

```bash
--edgelist <file>                   # Seed graph edges
--nodelist <file>                   # Seed graph nodes with years
--out-degree-bag <file>             # Out-degree distribution
--recency-probabilities <file>      # Temporal decay probabilities
--growth-rate <float>               # Network growth rate (e.g., 0.05)
--num-cycles <int>                  # Simulation duration (years)
--same-year-proportion <float>      # Same-year citations fraction
--output-file <file>                # Output edgelist
--auxiliary-information-file <file> # Output node attributes
--log-file <file>                   # Simulation log
--num-processors <int>              # OpenMP threads
```

## Typical Workflow

### 1. Model Selection
```
Real citation behavior?     → PA model
Null model/baseline?        → ER model  
Classical random graph?     → ER-GNP model
```

### 2. Parameter Tuning

See [PARAMETERS.md](PARAMETERS.md) for detailed guidance.

**PA Model:**
- Balance weights to match real data
- Adjust alpha for clustering
- Tune recency for temporal patterns

**ER Models:**
- Fixed-k: Use realistic out-degree distribution
- G(n,p): Choose p for target density

### 3. Running Simulations
```bash
# Quick test
./test.sh

# Full comparison
./examples/run_all_models.sh

# Custom run
./citation_models --model pa [parameters...]
```

### 4. Analysis
```bash
# Python analysis
python3 examples/analyze_networks.py

# Or manual analysis
import networkx as nx
G = nx.read_edgelist('results/pa_network.txt', create_using=nx.DiGraph())
```

## Key Outputs

### Network File
Tab-separated edgelist:
```
1    2
1    3
5    1
```

### Attributes File  
Node properties:
```
node_id  year  type   in_degree  out_degree  fitness  ...
1        2000  seed   5          2           67       ...
```

### Log File
Timestamped execution log:
```
[INFO][0-0:0:5](t=5s) current year is: 2001 and graph is 1200 nodes large
```

## Performance

### Build Time
- ~10 seconds on modern system

### Execution Time (rough estimates, 8 cores)
| Nodes | Cycles | PA Model | ER Models |
|-------|--------|----------|-----------|
| 1K    | 20     | <1 min   | <30 sec   |
| 10K   | 50     | 5-15 min | 2-5 min   |
| 100K  | 100    | 1-3 hrs  | 20-40 min |

### Memory Usage
- PA model: ~500 bytes/node
- ER models: ~200 bytes/node

## Example Results

After running `./examples/run_all_models.sh`:

```
results/
├── pa_network.txt              # PA model edges
├── er_network.txt              # ER fixed-k edges
├── er_gnp_network.txt          # ER G(n,p) edges
├── *_attributes.txt            # Node attributes
├── *_simulation.log            # Execution logs
├── degree_distributions.png    # Degree histograms
├── degree_ccdf.png             # CCDF comparison
├── metrics_comparison.png      # Bar chart comparison
└── network_metrics.csv         # Quantitative metrics
```

## Citation Analysis Use Cases

### 1. Model Validation
Compare simulated vs. real citation networks:
- Degree distributions
- Clustering coefficients
- Temporal patterns
- Citation lag distributions

### 2. Mechanism Testing
Identify which mechanisms drive observed patterns:
- Run PA with different weight combinations
- Compare against ER baseline
- Measure contribution of each factor

### 3. Prediction
Forecast network evolution:
- Fit parameters to historical data
- Project future growth
- Estimate future in-degrees

### 4. Null Model Testing
Test if patterns deviate from random:
- Generate ER baseline
- Compare real data statistics
- Statistical significance testing

## Troubleshooting

### Build Issues
```bash
# Clean rebuild
make clean && make

# Without OpenMP
make OMPFLAGS=""
```

### Runtime Issues
```bash
# Out of memory
--growth-rate 0.02 --num-cycles 10

# Slow execution
--num-processors 8 --log-level 0
```

### Analysis Issues
```bash
# Install Python dependencies
pip install pandas numpy matplotlib networkx seaborn
```

## Further Customization

### Adding Custom Models
1. Add model case in `abm.cpp::main()`
2. Implement citation logic
3. Update `main.cpp` help text

### Modifying Fitness Distribution
Edit `AssignPeakFitnessValues()` in `abm.h`

### Custom Analysis
See `examples/analyze_networks.py` as template

## Support

- Full docs: [README.md](README.md)
- Quick start: [QUICKSTART.md](QUICKSTART.md)
- Parameters: [PARAMETERS.md](PARAMETERS.md)
- Issues: Open GitHub issue

## References

- Barabási & Albert (1999). Emergence of scaling in random networks. *Science*
- Erdős & Rényi (1959). On random graphs. *Publicationes Mathematicae*
- Wang et al. (2013). Quantifying long-term scientific impact. *Science*

## License

MIT License - See LICENSE file

## Version

Version 1.0 - March 2025

---

**Ready to start?** Run `./test.sh` to verify installation!
