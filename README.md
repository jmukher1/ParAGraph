# Citation Network Growth Models

A C++ (and CUDA for GPU) implementation of citation network growth models including **Preferential Attachment (PA)** and **Erdős-Rényi (ER)** variants for analyzing scientific citation patterns for CPU and GPU.

The Repository is created to host all the code related to GPU based optimizations for Agent-based Modeling framework for Citation Graph growth.

The master branch is a fork from https://github.com/MinhyukPark/cpp_abm/blob/v5-p which is a c++ implementation (a baseline) to compare against for all the different variations.

There are 6 different branches:

1. Master: c++ implementation (a baseline) to compare against (logically similar to v5-p branch)
2. gpu: gpu implementation similar to (v5-parallel) implementing #1 in gpu
3. gpu-opt: Optimal version of the #2 (optimized gpu version)
4. cpu-model: c++ implementation #1 plus Erdos-Renyi model
5. gpu-model: gpu implementation #3 plus GPU optimized Erdos-Renyi model
6. mass_cuda: MASS_CUDA based mass_cuda based implemenation of Preferential Attachment model (similar to #1 and #2) and Erdos-Renyi model on GPU

This branch (cpu-model) contains a C++ implementation of citation network growth models including **Preferential Attachment (PA)** and **Erdős-Rényi (ER)** variants for analyzing scientific citation patterns.


## Models Implemented

### 1. Preferential Attachment (PA) - Default
The PA model simulates realistic citation behavior based on:
- **Preferential Attachment**: Popular papers get more citations (rich-get-richer)
- **Recency**: Recent papers are more likely to be cited
- **Fitness**: Papers have intrinsic quality affecting citation probability
- **Neighborhood**: Citations favor papers in local network neighborhoods

### 2. Erdős-Rényi Fixed-k (ER)
Each new node creates exactly k edges to existing nodes chosen uniformly at random.
- Baseline model for comparison
- No preferential attachment
- Controlled out-degree distribution

### 3. Erdős-Rényi G(n,p) (ER-GNP)
Each new node connects to each existing node independently with probability p.
- Classical random graph model
- Variable out-degree (binomial distribution)
- Stochastic edge formation

## Directory Structure

```
citation_models/
├── src/
│   ├── main.cpp          # Command-line interface
│   ├── abm.cpp           # Core simulation logic
│   └── graph.cpp         # Graph data structure
├── includes/
│   ├── abm.h             # ABM class definition
│   ├── graph.h           # Graph class definition
│   └── argparse.h        # Command-line parsing
├── build/                # Compiled object files
├── examples/             # Example input files and scripts
├── makefile              # Build configuration
└── README.md             # This file
```

## Installation

### Prerequisites
- C++ compiler with C++20 support (g++ 10+)
- OpenMP for parallelization
- Make

### Build
```bash
make clean
make
```

This creates the `citation_models` executable.

## Usage

### Basic Command Structure
```bash
./citation_models \
  --model <pa|er|er-gnp> \
  --edgelist <input_edges.txt> \
  --nodelist <input_nodes.txt> \
  --out-degree-bag <outdegree_dist.csv> \
  --recency-probabilities <recency_probs.csv> \
  --growth-rate <rate> \
  --num-cycles <cycles> \
  --same-year-proportion <proportion> \
  --output-file <output_edges.txt> \
  --auxiliary-information-file <output_attrs.txt> \
  --log-file <log.txt>
```

### Model-Specific Parameters

#### Preferential Attachment (--model pa)
```bash
./citation_models \
  --model pa \
  --preferential-weight 0.5 \
  --recency-weight 0.3 \
  --fitness-weight 0.2 \
  --alpha 0.7 \
  --fully-random-citations 0.05 \
  --edgelist seed_graph.txt \
  --nodelist seed_nodes.txt \
  --out-degree-bag outdegree.csv \
  --recency-probabilities recency.csv \
  --growth-rate 0.05 \
  --num-cycles 20 \
  --same-year-proportion 0.1 \
  --output-file output_pa.txt \
  --auxiliary-information-file output_pa_attrs.txt \
  --log-file pa_simulation.log \
  --num-processors 8
```

**PA Parameters:**
- `--preferential-weight`: Weight for degree-based attachment (default: -1 = random)
- `--recency-weight`: Weight for temporal recency (default: -1 = random)
- `--fitness-weight`: Weight for intrinsic fitness (default: -1 = random)
- `--alpha`: Proportion of citations from 1-hop vs 2-hop neighborhood (default: -1 = random)
- `--fully-random-citations`: Fraction of completely random citations (default: 0.05)

#### Erdős-Rényi Fixed-k (--model er)
```bash
./citation_models \
  --model er \
  --edgelist seed_graph.txt \
  --nodelist seed_nodes.txt \
  --out-degree-bag outdegree.csv \
  --recency-probabilities recency.csv \
  --growth-rate 0.05 \
  --num-cycles 20 \
  --same-year-proportion 0.1 \
  --output-file output_er.txt \
  --auxiliary-information-file output_er_attrs.txt \
  --log-file er_simulation.log \
  --num-processors 8
```

**ER Fixed-k:** Uses the out-degree bag to determine k for each new node.

#### Erdős-Rényi G(n,p) (--model er-gnp)
```bash
./citation_models \
  --model er-gnp \
  --er-probability 0.01 \
  --edgelist seed_graph.txt \
  --nodelist seed_nodes.txt \
  --out-degree-bag outdegree.csv \
  --recency-probabilities recency.csv \
  --growth-rate 0.05 \
  --num-cycles 20 \
  --same-year-proportion 0.1 \
  --output-file output_er_gnp.txt \
  --auxiliary-information-file output_er_gnp_attrs.txt \
  --log-file er_gnp_simulation.log \
  --num-processors 8
```

**ER-GNP Parameters:**
- `--er-probability`: Edge probability p (default: 0.01)

### Common Parameters

- `--edgelist`: Input edgelist file (tab/comma/space separated: source, target)
- `--nodelist`: Input nodelist file (node_id, year)
- `--out-degree-bag`: CSV file with out-degree distribution (index, degree)
- `--recency-probabilities`: CSV file with recency probabilities (year_diff, probability)
- `--planted-nodes`: Optional CSV for special high-fitness nodes (year, lag, peak_value, peak_duration, count)
- `--growth-rate`: Fraction of nodes added per cycle (e.g., 0.05 = 5% growth)
- `--num-cycles`: Number of time steps to simulate
- `--same-year-proportion`: Fraction of same-year citations
- `--output-file`: Output edgelist file
- `--auxiliary-information-file`: Output node attributes file
- `--log-file`: Simulation log file
- `--num-processors`: Number of OpenMP threads (default: 1)
- `--log-level`: 0=silent, 1=info, 2=verbose (default: 1)

## Input File Formats

### 1. Edgelist (seed_graph.txt)
```
# source target
1 2
1 3
2 4
```

### 2. Nodelist (seed_nodes.txt)
```
# node_id year
1 2000
2 2000
3 2001
4 2001
```

### 3. Out-Degree Bag (outdegree.csv)
```
# index,degree
0,5
1,3
2,7
3,2
```

### 4. Recency Probabilities (recency.csv)
```
# year_diff,probability
0,0.3
1,0.25
2,0.2
3,0.15
4,0.1
```

### 5. Planted Nodes (optional, planted_nodes.csv)
```
# year,fitness_lag,fitness_peak,peak_duration,count
1,0,1000,1000,5
5,0,500,1000,10
```

## Output Files

### 1. Output Edgelist
Tab-separated edgelist of the grown network:
```
1	2
1	3
2	4
5	1
5	3
```

### 2. Auxiliary Information File
Node attributes including year, type, degrees, fitness values, and model-specific parameters:
```
node_id	year	type	in_degree	out_degree	fitness_peak_value	...
1	2000	seed	3	2	45	...
2	2000	seed	1	1	23	...
5	2001	agent	0	5	67	...
```

### 3. Log File
Detailed simulation log with timestamps:
```
[INFO][0-0:0:1](t=1s) loaded graph
[INFO][0-0:0:1](t=1s) initialized fitness for the seed graph
[INFO][0-0:0:2](t=2s) current year is: 2001 and the graph is 1000 nodes large
...
```

## Example Workflows

### Compare PA vs ER Models
```bash
# Run PA model
./citation_models --model pa \
  --edgelist data/seed.txt --nodelist data/nodes.txt \
  --out-degree-bag data/outdeg.csv --recency-probabilities data/recency.csv \
  --growth-rate 0.05 --num-cycles 50 --same-year-proportion 0.1 \
  --output-file results/pa_network.txt \
  --auxiliary-information-file results/pa_attrs.txt \
  --log-file results/pa.log --num-processors 8

# Run ER fixed-k model
./citation_models --model er \
  --edgelist data/seed.txt --nodelist data/nodes.txt \
  --out-degree-bag data/outdeg.csv --recency-probabilities data/recency.csv \
  --growth-rate 0.05 --num-cycles 50 --same-year-proportion 0.1 \
  --output-file results/er_network.txt \
  --auxiliary-information-file results/er_attrs.txt \
  --log-file results/er.log --num-processors 8

# Run ER G(n,p) model
./citation_models --model er-gnp --er-probability 0.005 \
  --edgelist data/seed.txt --nodelist data/nodes.txt \
  --out-degree-bag data/outdeg.csv --recency-probabilities data/recency.csv \
  --growth-rate 0.05 --num-cycles 50 --same-year-proportion 0.1 \
  --output-file results/er_gnp_network.txt \
  --auxiliary-information-file results/er_gnp_attrs.txt \
  --log-file results/er_gnp.log --num-processors 8
```

### Analysis with Python
```python
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load networks
pa_edges = pd.read_csv('results/pa_network.txt', sep='\t', names=['source', 'target'])
er_edges = pd.read_csv('results/er_network.txt', sep='\t', names=['source', 'target'])

# Create graphs
G_pa = nx.from_pandas_edgelist(pa_edges, 'source', 'target', create_using=nx.DiGraph())
G_er = nx.from_pandas_edgelist(er_edges, 'source', 'target', create_using=nx.DiGraph())

# Compare in-degree distributions
in_deg_pa = [d for n, d in G_pa.in_degree()]
in_deg_er = [d for n, d in G_er.in_degree()]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(in_deg_pa, bins=50, alpha=0.7, label='PA')
plt.xlabel('In-degree')
plt.ylabel('Frequency')
plt.title('PA Model')
plt.yscale('log')

plt.subplot(1, 2, 2)
plt.hist(in_deg_er, bins=50, alpha=0.7, label='ER', color='orange')
plt.xlabel('In-degree')
plt.ylabel('Frequency')
plt.title('ER Model')
plt.yscale('log')
plt.tight_layout()
plt.savefig('degree_comparison.png', dpi=300)
```

## Performance Optimization

### Parallel Execution
The code uses OpenMP for parallel execution. Set the number of processors:
```bash
./citation_models --num-processors 16 ...
```

### Memory Considerations
For large networks (>1M nodes), ensure sufficient RAM:
- PA model: ~500 bytes per node
- ER models: ~200 bytes per node

## Model Details

### Preferential Attachment Mechanism
The PA model computes citation probability as:
```
P(cite node i) ∝ (degree(i)^γ + 1)^w_pa × recency(i)^w_rec × fitness(i)^w_fit
```

Where:
- γ = 3 (gamma parameter for degree)
- w_pa, w_rec, w_fit are normalized weights
- Weighted random sampling with Efraimidis-Spirakis algorithm

### Erdős-Rényi Mechanisms

**Fixed-k**: For each new node with assigned out-degree k:
```
citations = sample k nodes uniformly from existing nodes
```

**G(n,p)**: For each new node:
```
for each existing node i:
    if random() < p:
        create edge to i
```

## Troubleshooting

### Compilation Errors
```bash
# If OpenMP not found
make OMPFLAGS=""

# If C++20 not supported
# Edit makefile: change -std=c++20 to -std=c++17
```

### Runtime Issues

**Out of memory:**
- Reduce `--num-cycles` or `--growth-rate`
- Use ER models (lower memory footprint)

**Slow execution:**
- Increase `--num-processors`
- Use `--log-level 0` to disable verbose logging
- For ER models, ensure `--er-probability` is not too high

## Citation

If you use this code in your research, please cite:

```bibtex
@software{citation_models_2025,
  title={Citation Network Growth Models: PA and ER Variants},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/citation_models}
}
```

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].

## Acknowledgments

- Based on agent-based modeling framework from UIUC
- Implements algorithms from https://github.com/MinhyukPark/cpp_abm/blob/v5-p, Efraimidis-Spirakis (WRS)


