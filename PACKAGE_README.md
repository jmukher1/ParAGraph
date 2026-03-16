# Citation Network Growth Models - Complete Package

## 🎯 What's Included

This package provides a complete C++ implementation of citation network growth models with command-line model selection:

### Three Models Implemented:
1. ✅ **Preferential Attachment (PA)** - Realistic citation dynamics
2. ✅ **Erdős-Rényi Fixed-k (ER)** - Uniform random baseline
3. ✅ **Erdős-Rényi G(n,p) (ER-GNP)** - Probabilistic edge formation

### Key Features:
- ✅ Command-line model selection (--model pa|er|er-gnp)
- ✅ Complete parameter control for each model
- ✅ OpenMP parallelization for performance
- ✅ Comprehensive example data and scripts
- ✅ Python analysis tools included
- ✅ Full documentation and guides

## 📦 Package Contents

```
citation_models_complete.tar.gz (293 KB)
└── citation_models/
    ├── INSTALL.md              ⭐ Start here - Installation guide
    ├── QUICKSTART.md           🚀 5-minute quick start
    ├── README.md               📖 Complete documentation
    ├── SUMMARY.md              📋 Project overview
    ├── PARAMETERS.md           🎛️ Parameter tuning guide
    ├── LICENSE                 ⚖️ MIT License
    │
    ├── src/                    💻 Source code
    │   ├── main.cpp           # CLI with model selection
    │   ├── abm.cpp            # PA + ER + ER-GNP implementations
    │   └── graph.cpp          # Graph data structure
    │
    ├── includes/               📄 Headers
    │   ├── abm.h              # ABM class
    │   ├── graph.h            # Graph class
    │   └── argparse.h         # CLI parsing
    │
    ├── examples/               📊 Examples & analysis
    │   ├── seed_edgelist.txt
    │   ├── seed_nodelist.txt
    │   ├── outdegree_bag.csv
    │   ├── recency_probabilities.csv
    │   ├── run_all_models.sh  # Run all 3 models
    │   └── analyze_networks.py # Python analysis
    │
    ├── makefile                🔨 Build configuration
    └── test.sh                 ✅ Quick verification
```

## 🚀 Quick Start (3 Steps)

### 1. Extract and Build
```bash
tar -xzf citation_models_complete.tar.gz
cd citation_models
make
```

### 2. Test
```bash
./test.sh
```

### 3. Run Examples
```bash
./examples/run_all_models.sh
python3 examples/analyze_networks.py  # Optional
```

**Done!** Results in `results/` directory.

## 📚 Documentation Guide

Read in this order:

1. **[INSTALL.md](INSTALL.md)** - Installation & troubleshooting
2. **[QUICKSTART.md](QUICKSTART.md)** - Get running in 5 minutes
3. **[SUMMARY.md](SUMMARY.md)** - Project overview
4. **[README.md](README.md)** - Complete documentation
5. **[PARAMETERS.md](PARAMETERS.md)** - Tuning guide

## 🎮 Usage Examples

### Run Preferential Attachment Model
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
  --output-file pa_network.txt \
  --auxiliary-information-file pa_attrs.txt \
  --log-file pa.log \
  --num-processors 4
```

### Run Erdős-Rényi Fixed-k Model
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
  --output-file er_network.txt \
  --auxiliary-information-file er_attrs.txt \
  --log-file er.log \
  --num-processors 4
```

### Run Erdős-Rényi G(n,p) Model
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
  --output-file er_gnp_network.txt \
  --auxiliary-information-file er_gnp_attrs.txt \
  --log-file er_gnp.log \
  --num-processors 4
```

## 🔬 Model Comparison

| Feature | PA Model | ER Fixed-k | ER G(n,p) |
|---------|----------|------------|-----------|
| **Realism** | High | Low | Low |
| **Preferential Attachment** | ✅ | ❌ | ❌ |
| **Recency Effects** | ✅ | ❌ | ❌ |
| **Fitness/Quality** | ✅ | ❌ | ❌ |
| **Use Case** | Realistic simulation | Null model | Theoretical baseline |
| **Degree Distribution** | Power-law | Controlled | Binomial |
| **Computation** | Slower | Faster | Faster |

## 📊 Input File Formats

### Edgelist (tab/comma/space separated)
```
1	2
1	3
2	4
```

### Nodelist (node_id, year)
```
1	2000
2	2000
3	2001
```

### Out-degree Bag (index, degree)
```
0,5
1,3
2,7
```

### Recency Probabilities (year_diff, probability)
```
0,0.30
1,0.25
2,0.20
```

## 📈 Output Files

### Network File
```
source	target
1	2
5	3
```

### Attributes File
```
node_id	year	type	in_degree	out_degree	fitness...
1	2000	seed	3	2	45...
```

### Log File
```
[INFO][0-0:0:1](t=1s) loaded graph
[INFO][0-0:0:2](t=2s) current year is: 2001...
```

## 🛠️ System Requirements

**Minimum:**
- Linux/macOS/Windows (WSL)
- g++ 10.0+ with C++20 support
- 4 GB RAM
- 1 GB disk space

**Recommended:**
- Multi-core CPU (4+ cores)
- 8+ GB RAM
- OpenMP support

**Optional:**
- Python 3.7+ (for analysis)
- pandas, numpy, matplotlib, networkx, seaborn

## ✅ Verification

After installation, you should see:

```bash
$ ./test.sh
==========================================
Citation Models - Comprehensive Test
==========================================
...
Test 1/3: Preferential Attachment Model
  ✓ PA model completed successfully
Test 2/3: Erdős-Rényi Fixed-k Model
  ✓ ER fixed-k model completed successfully
Test 3/3: Erdős-Rényi G(n,p) Model
  ✓ ER G(n,p) model completed successfully
==========================================
All Tests Passed Successfully!
==========================================
```

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@software{citation_models_2025,
  title={Citation Network Growth Models: Preferential Attachment and Erdős-Rényi Variants},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/citation_models}
}
```

## 📄 License

MIT License - Free to use, modify, and distribute.

## 🤝 Support

- 📖 Read the documentation (INSTALL.md, QUICKSTART.md, README.md)
- 🐛 Report issues on GitHub
- 💡 Suggest features or improvements
- 🤝 Contribute via pull requests

## 🗺️ Roadmap

Future enhancements:
- [ ] GPU acceleration (CUDA implementation)
- [ ] Additional models (small-world, community structure)
- [ ] Web interface for visualization
- [ ] More analysis tools
- [ ] Docker container

## 📞 Contact

For questions or support:
- Open an issue on GitHub
- Check documentation first
- Review examples

---

## 🎉 Ready to Start!

1. Extract: `tar -xzf citation_models_complete.tar.gz`
2. Build: `cd citation_models && make`
3. Test: `./test.sh`
4. Run: `./examples/run_all_models.sh`
5. Analyze: `python3 examples/analyze_networks.py`

**Happy simulating! 🚀**
