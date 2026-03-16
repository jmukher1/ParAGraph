# Installation and Verification Guide

## Package Contents

The `citation_models_complete.tar.gz` package contains:

```
citation_models/
├── README.md                        # Complete documentation
├── QUICKSTART.md                    # 5-minute quick start
├── SUMMARY.md                       # Project overview
├── PARAMETERS.md                    # Parameter tuning guide
├── LICENSE                          # MIT License
├── makefile                         # Build configuration
├── test.sh                          # Quick test script
│
├── src/                            # Source code
│   ├── main.cpp                    # CLI entry point
│   ├── abm.cpp                     # Core simulation (with PA, ER, ER-GNP)
│   └── graph.cpp                   # Graph data structure
│
├── includes/                       # Headers
│   ├── abm.h                       # ABM class with model selection
│   ├── graph.h                     # Graph class
│   └── argparse.h                  # Command-line parsing
│
└── examples/                       # Examples and scripts
    ├── seed_edgelist.txt          # Example seed graph
    ├── seed_nodelist.txt          # Example node list
    ├── outdegree_bag.csv          # Example degree distribution
    ├── recency_probabilities.csv  # Example recency weights
    ├── run_all_models.sh          # Run all 3 models
    └── analyze_networks.py        # Python analysis script
```

## Installation Steps

### 1. Extract the Package

```bash
tar -xzf citation_models_complete.tar.gz
cd citation_models
```

### 2. Verify Prerequisites

**Required:**
- C++ compiler with C++20 support (g++ 10 or later)
- OpenMP
- Make

**Check your setup:**
```bash
g++ --version    # Should be 10.0 or higher
make --version   # Any recent version
```

**Optional (for analysis):**
- Python 3.7+
- Python packages: pandas, numpy, matplotlib, networkx, seaborn

### 3. Build the Project

```bash
make clean
make
```

**Expected output:**
```
mkdir -p build
g++ -O3 -std=c++20 -fopenmp -c -I./includes src/main.cpp -o build/main.o
g++ -O3 -std=c++20 -fopenmp -c -I./includes src/abm.cpp -o build/abm.o
g++ -O3 -std=c++20 -c -I./includes src/graph.cpp -o build/graph.o
g++ -O3 -std=c++20 -fopenmp build/main.o build/abm.o build/graph.o -I./includes -lpthread -o citation_models -lstdc++
```

**If successful:** You'll see the `citation_models` executable.

```bash
ls -lh citation_models
# Expected: -rwxr-xr-x ... citation_models
```

### 4. Run Quick Test

```bash
./test.sh
```

**Expected output:**
```
==========================================
Citation Models - Comprehensive Test
==========================================

Test Configuration:
  Growth rate: 0.1
  Num cycles: 5
  Processors: 2

Test 1/3: Preferential Attachment Model
  Running...
  ✓ PA model completed successfully
    Output: XXX edges generated

Test 2/3: Erdős-Rényi Fixed-k Model
  Running...
  ✓ ER fixed-k model completed successfully
    Output: XXX edges generated

Test 3/3: Erdős-Rényi G(n,p) Model
  Running...
  ✓ ER G(n,p) model completed successfully
    Output: XXX edges generated

==========================================
All Tests Passed Successfully!
==========================================
```

### 5. Run Full Examples

```bash
./examples/run_all_models.sh
```

This will:
- Run PA model (realistic citations)
- Run ER fixed-k model (uniform random baseline)
- Run ER-GNP model (probabilistic edges)
- Generate output in `results/` directory

### 6. Analyze Results (Optional)

If you have Python installed:

```bash
python3 examples/analyze_networks.py
```

This generates:
- `results/degree_distributions.png`
- `results/degree_ccdf.png`
- `results/metrics_comparison.png`
- `results/network_metrics.csv`

## Troubleshooting

### Build Errors

**Problem: "g++: command not found"**
```bash
# Ubuntu/Debian
sudo apt-get install g++ make

# macOS
xcode-select --install

# CentOS/RHEL
sudo yum install gcc-c++ make
```

**Problem: "c++20: No such file or directory"**
```bash
# Check g++ version
g++ --version

# If < 10.0, update or use C++17
# Edit makefile: change -std=c++20 to -std=c++17
```

**Problem: OpenMP not found**
```bash
# Build without OpenMP (single-threaded)
make clean
make OMPFLAGS=""
```

### Runtime Errors

**Problem: "Cannot open file"**
```bash
# Verify example files exist
ls examples/

# Check you're in the citation_models directory
pwd  # Should end with /citation_models
```

**Problem: "Segmentation fault"**
```bash
# This shouldn't happen with the examples
# If it does, try with smaller parameters
./citation_models --growth-rate 0.05 --num-cycles 5 ...
```

### Analysis Errors

**Problem: "ModuleNotFoundError: No module named 'pandas'"**
```bash
# Install Python dependencies
pip install pandas numpy matplotlib networkx seaborn

# Or with conda
conda install pandas numpy matplotlib networkx seaborn
```

## Verification Checklist

- [ ] Package extracted successfully
- [ ] `make` completed without errors
- [ ] `citation_models` executable created
- [ ] `./test.sh` passed all tests
- [ ] Output files created in `test_results/`
- [ ] `./examples/run_all_models.sh` completed
- [ ] Results generated in `results/` directory

## Quick Command Reference

### Build Commands
```bash
make              # Build project
make clean        # Remove compiled files
make clean && make # Clean rebuild
```

### Test Commands
```bash
./test.sh                          # Quick test (3 models)
./examples/run_all_models.sh      # Full examples
./citation_models --help           # Show usage
```

### Run Individual Models
```bash
# Preferential Attachment
./citation_models --model pa [options...]

# Erdős-Rényi Fixed-k
./citation_models --model er [options...]

# Erdős-Rényi G(n,p)
./citation_models --model er-gnp --er-probability 0.01 [options...]
```

### Analysis Commands
```bash
python3 examples/analyze_networks.py   # Generate plots
head -20 results/pa_network.txt        # View edges
wc -l results/*.txt                    # Count edges
```

## Next Steps

1. **Read the documentation:**
   - Start with [QUICKSTART.md](QUICKSTART.md) for immediate usage
   - Read [README.md](README.md) for complete documentation
   - Check [PARAMETERS.md](PARAMETERS.md) for parameter tuning

2. **Try your own data:**
   - Format your seed graph as edgelist
   - Create nodelist with years
   - Prepare out-degree distribution
   - Run simulations

3. **Compare models:**
   - Run all three models with same parameters
   - Compare degree distributions
   - Analyze temporal patterns

4. **Customize:**
   - Adjust model parameters
   - Modify fitness distributions
   - Add custom analysis

## Getting Help

If you encounter issues:

1. Check the documentation files (README, QUICKSTART, PARAMETERS)
2. Verify all prerequisites are installed
3. Try the examples first before custom data
4. Check error messages in log files

## Example Session

Complete workflow from installation to analysis:

```bash
# 1. Extract and build
tar -xzf citation_models_complete.tar.gz
cd citation_models
make

# 2. Quick test
./test.sh

# 3. Run examples
./examples/run_all_models.sh

# 4. Analyze (if Python available)
python3 examples/analyze_networks.py

# 5. View results
ls -lh results/
head results/pa_network.txt

# 6. Run custom simulation
./citation_models --model pa \
  --preferential-weight 0.6 \
  --recency-weight 0.3 \
  --fitness-weight 0.1 \
  --alpha 0.8 \
  --edgelist examples/seed_edgelist.txt \
  --nodelist examples/seed_nodelist.txt \
  --out-degree-bag examples/outdegree_bag.csv \
  --recency-probabilities examples/recency_probabilities.csv \
  --growth-rate 0.05 \
  --num-cycles 30 \
  --same-year-proportion 0.1 \
  --output-file my_custom_network.txt \
  --auxiliary-information-file my_custom_attrs.txt \
  --log-file my_custom.log \
  --num-processors 4
```

## Success Criteria

You know everything is working when:

1. ✓ Build completes without errors
2. ✓ Test script passes all tests
3. ✓ Example run generates output files
4. ✓ Output files are non-empty and well-formed
5. ✓ Log files show successful completion

## Contact & Support

For questions, issues, or contributions:
- Check documentation first
- Review examples
- Open issue on GitHub (if applicable)

---

**Installation complete!** You're ready to start simulating citation networks.
