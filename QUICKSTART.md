# Quick Start Guide

## 1. Build the Project

```bash
cd citation_models
make clean
make
```

You should see:
```
mkdir -p build
g++ -O3 -std=c++20  -c -I./includes src/graph.cpp -o build/graph.o
g++ -O3 -std=c++20  -fopenmp -c -I./includes src/abm.cpp -o build/abm.o
g++ -O3 -std=c++20  -fopenmp -c -I./includes src/main.cpp -o build/main.o
g++ -O3 -std=c++20  -fopenmp build/main.o build/abm.o build/graph.o -I./includes -lpthread -o citation_models -lstdc++
```

## 2. Test with Example Data

Run all three models with example data:

```bash
cd citation_models
./examples/run_all_models.sh
```

This will:
- Run the PA (Preferential Attachment) model
- Run the ER (Erdős-Rényi fixed-k) model  
- Run the ER-GNP (Erdős-Rényi G(n,p)) model
- Save results to `results/` directory

## 3. Analyze Results

View the output files:

```bash
# View network sizes
wc -l results/*_network.txt

# View first few edges
head -20 results/pa_network.txt

# View node attributes
head -20 results/pa_attributes.txt
```

Or use the Python analysis script:

```bash
python3 examples/analyze_networks.py
```

This generates:
- `results/degree_distributions.png` - Degree distribution histograms
- `results/degree_ccdf.png` - Complementary CDF comparison
- `results/metrics_comparison.png` - Key metrics comparison
- `results/network_metrics.csv` - Quantitative metrics

## 4. Run Individual Models

### Preferential Attachment (PA)

```bash
./citation_models \
  --model pa \
  --preferential-weight 0.5 \
  --recency-weight 0.3 \
  --fitness-weight 0.2 \
  --alpha 0.7 \
  --edgelist examples/seed_edgelist.txt \
  --nodelist examples/seed_nodelist.txt \
  --out-degree-bag examples/outdegree_bag.csv \
  --recency-probabilities examples/recency_probabilities.csv \
  --growth-rate 0.05 \
  --num-cycles 20 \
  --same-year-proportion 0.1 \
  --output-file my_pa_network.txt \
  --auxiliary-information-file my_pa_attrs.txt \
  --log-file my_pa.log \
  --num-processors 4
```

### Erdős-Rényi Fixed-k (ER)

```bash
./citation_models \
  --model er \
  --edgelist examples/seed_edgelist.txt \
  --nodelist examples/seed_nodelist.txt \
  --out-degree-bag examples/outdegree_bag.csv \
  --recency-probabilities examples/recency_probabilities.csv \
  --growth-rate 0.05 \
  --num-cycles 20 \
  --same-year-proportion 0.1 \
  --output-file my_er_network.txt \
  --auxiliary-information-file my_er_attrs.txt \
  --log-file my_er.log \
  --num-processors 4
```

### Erdős-Rényi G(n,p) (ER-GNP)

```bash
./citation_models \
  --model er-gnp \
  --er-probability 0.01 \
  --edgelist examples/seed_edgelist.txt \
  --nodelist examples/seed_nodelist.txt \
  --out-degree-bag examples/outdegree_bag.csv \
  --recency-probabilities examples/recency_probabilities.csv \
  --growth-rate 0.05 \
  --num-cycles 20 \
  --same-year-proportion 0.1 \
  --output-file my_er_gnp_network.txt \
  --auxiliary-information-file my_er_gnp_attrs.txt \
  --log-file my_er_gnp.log \
  --num-processors 4
```

## 5. Key Parameters to Adjust

### Growth Parameters (All Models)
- `--growth-rate`: How fast the network grows (e.g., 0.05 = 5% per cycle)
- `--num-cycles`: How many time steps to simulate
- `--same-year-proportion`: Fraction of same-year citations

### PA Model Specific
- `--preferential-weight`: Weight for degree-based attachment (0-1)
- `--recency-weight`: Weight for temporal recency (0-1)
- `--fitness-weight`: Weight for intrinsic quality (0-1)
- `--alpha`: Proportion from 1-hop vs 2-hop neighborhood (0-1)

### ER-GNP Model Specific
- `--er-probability`: Edge probability p (typical range: 0.001-0.05)

### Performance
- `--num-processors`: Number of CPU cores to use
- `--log-level`: 0 (silent), 1 (info), 2 (verbose)

## 6. Understanding Output Files

### Network File (e.g., `pa_network.txt`)
Tab-separated edgelist:
```
source  target
1       2
1       3
5       1
```

### Attributes File (e.g., `pa_attributes.txt`)
Tab-separated node attributes:
```
node_id  year  type   in_degree  out_degree  fitness_peak_value  ...
1        2000  seed   3          2           45                  ...
5        2001  agent  0          5           67                  ...
```

### Log File (e.g., `pa.log`)
Detailed timestamped simulation log:
```
[INFO][0-0:0:1](t=1s) loaded graph
[INFO][0-0:0:2](t=2s) current year is: 2001 and the graph is 1000 nodes large
```

## 7. Typical Workflow

1. **Prepare your data** in the required format (see README.md)
2. **Run simulations** for each model you want to compare
3. **Analyze results** with Python or your preferred tool
4. **Compare metrics** like degree distributions, clustering, etc.
5. **Iterate** on parameters to match your target properties

## 8. Troubleshooting

**"Command not found"**
```bash
# Make sure you built the project
make clean && make
```

**"Cannot open file"**
```bash
# Check file paths are correct
ls examples/
```

**"Out of memory"**
```bash
# Reduce growth rate or number of cycles
--growth-rate 0.02 --num-cycles 10
```

**Slow execution**
```bash
# Increase processors and reduce logging
--num-processors 8 --log-level 0
```

## 9. Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Modify example data files for your specific use case
- Implement custom analysis in Python using NetworkX
- Try different parameter combinations to match real citation data

## Need Help?

Check the full README.md or open an issue on GitHub.
