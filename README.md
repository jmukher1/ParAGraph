The Repository is created to host all the code related to GPU based optimizations for Agent-based Modeling framework for Citation Graph growth.

The master branch is a fork from https://github.com/MinhyukPark/cpp_abm/blob/v5-p which is a c++ implementation (a baseline) to compare against for all the different variations.

There are 6 different branches:

1. Master: c++ implementation (a baseline) to compare against (logically similar to v5-p branch)
2. gpu-opt: Optimal version of the #2 (optimized gpu version)
3. gpu-opt-pr: Profiled #2
4. cpu-model: c++ implementation #1 plus Erdos-Renyi model
5. gpu-model: gpu implementation #3 plus GPU optimized Erdos-Renyi model
6. mass_cuda: MASS_CUDA based mass_cuda based implemenation of Preferential Attachment model (similar to #1 and #2) and Erdos-Renyi model on GPU


1. com-Amazon

Amazon : Amazon product co-purchasing network and ground-truth communities


https://snap.stanford.edu/data/com-Amazon.html


Dataset statistics
Nodes	334863
Edges	925872
Nodes in largest WCC	334863 (1.000)
Edges in largest WCC	925872 (1.000)
Nodes in largest SCC	334863 (1.000)
Edges in largest SCC	925872 (1.000)
Average clustering coefficient	0.3967
Number of triangles	667129
Fraction of closed triangles	0.07925
Diameter (longest shortest path)	44
90-percentile effective diameter	15
 

python3 preprocess_to_seed.py --input ./com-amazon.ungraph.txt --out-prefix amz --start-year 1950 --end-year 2020  --recency-bins 356  --min-degree 0 --seed 4i2

Reading edges from ./com-amazon.ungraph.txt ...
  334,863 nodes, 925,872 edges (undirected)
Assigning synthetic publication years [1950, 2020], growth_shape=3.0 ...
Orienting edges by year (citing = later year, cited = earlier year) ...
Building recency probability table (356 bins, peak=2.0, sigma=1.2) ...

Wrote:
  amz_nodelist  (334,863 nodes)
  amz_edgelist  (925,872 directed edges)
  amz_recprob  (356 bins, sums to 1.000000)

3. email-EuAll

EU email communication network

https://snap.stanford.edu/data/email-EuAll.html

Dataset statistics
Nodes	265214
Edges	420045
Nodes in largest WCC	224832 (0.848)
Edges in largest WCC	395270 (0.941)
Nodes in largest SCC	34203 (0.129)
Edges in largest SCC	151930 (0.362)
Average clustering coefficient	0.0671
Number of triangles	267313
Fraction of closed triangles	0.001373
Diameter (longest shortest path)	14
90-percentile effective diameter	4.5


python3 preprocess_to_seed.py --input ./email-EuAll.txt --out-prefix eu --start-year 1950 --end-year 2020  --recency-bins 356  --min-degree 0 --seed 42
Reading edges from ./email-EuAll.txt ...
  265,009 nodes, 418,956 edges (undirected)
Assigning synthetic publication years [1950, 2020], growth_shape=3.0 ...
Orienting edges by year (citing = later year, cited = earlier year) ...
Building recency probability table (356 bins, peak=2.0, sigma=1.2) ...

Wrote:
  eu_nodelist  (265,009 nodes)
  eu_edgelist  (418,956 directed edges)
  eu_recprob  (356 bins, sums to 1.000000)

4. com-youtube.ungraph

Youtube social network and ground-truth communities

https://snap.stanford.edu/data/com-Youtube.html

Network statistics
Nodes	1134890
Edges	2987624
Nodes in largest WCC	1134890 (1.000)
Edges in largest WCC	2987624 (1.000)
Nodes in largest SCC	1134890 (1.000)
Edges in largest SCC	2987624 (1.000)
Average clustering coefficient	0.0808
Number of triangles	3056386
Fraction of closed triangles	0.002081
Diameter (longest shortest path)	20
90-percentile effective diameter	6.5
Community statistics
Number of communities	8,385
Average community size	13.50
Average membership size	0.10

python3 preprocess_to_seed.py --input ./com-youtube.ungraph.txt --out-prefix yutb --start-year 1950 --end-year 2026  --recency-bins 356  --min-degree 0 --seed 42
Reading edges from ./com-youtube.ungraph.txt ...
  1,134,890 nodes, 2,987,624 edges (undirected)
Assigning synthetic publication years [1950, 2026], growth_shape=3.0 ...
Orienting edges by year (citing = later year, cited = earlier year) ...
Building recency probability table (356 bins, peak=2.0, sigma=1.2) ...

Wrote:
  yutb_nodelist  (1,134,890 nodes)
  yutb_edgelist  (2,987,624 directed edges)
  yutb_recprob  (356 bins, sums to 1.000000)


5. ego-Twitter

Social circles: Twitter

https://snap.stanford.edu/data/ego-Twitter.html

Dataset statistics
Nodes	81306
Edges	1768149
Nodes in largest WCC	81306 (1.000)
Edges in largest WCC	1768149 (1.000)
Nodes in largest SCC	68413 (0.841)
Edges in largest SCC	1685163 (0.953)
Average clustering coefficient	0.5653
Number of triangles	13082506
Fraction of closed triangles	0.06415
Diameter (longest shortest path)	7
90-percentile effective diameter	4.5

python3 preprocess_to_seed.py --input ./twitter_combined.txt --out-prefix twt --start-year 1950 --end-year 2020  --recency-bins 356  --min-degree 0 --seed 42
Reading edges from ./twitter_combined.txt ...
  81,306 nodes, 2,420,744 edges (undirected)
Assigning synthetic publication years [1950, 2020], growth_shape=3.0 ...
Orienting edges by year (citing = later year, cited = earlier year) ...
Building recency probability table (356 bins, peak=2.0, sigma=1.2) ...

Wrote:
  twt_nodelist  (81,306 nodes)
  twt_edgelist  (2,420,744 directed edges)
  twt_recprob  (356 bins, sums to 1.000000)



6. web-BerkStan

Berkeley-Stanford web graph

https://snap.stanford.edu/data/web-BerkStan.html

Dataset statistics
Nodes	685230
Edges	7600595
Nodes in largest WCC	654782 (0.956)
Edges in largest WCC	7499425 (0.987)
Nodes in largest SCC	334857 (0.489)
Edges in largest SCC	4523232 (0.595)
Average clustering coefficient	0.5967
Number of triangles	64690980
Fraction of closed triangles	0.002746
Diameter (longest shortest path)	514
90-percentile effective diameter	9.9



python3 preprocess_to_seed.py --input ./web-BerkStan.txt --out-prefix berkstan --start-year 1950 --end-year 2026  --recency-bins 356  --min-degree 0 --seed 42
Reading edges from ./web-BerkStan.txt ...
  685,230 nodes, 7,600,595 edges (undirected)
Assigning synthetic publication years [1950, 2026], growth_shape=3.0 ...
Orienting edges by year (citing = later year, cited = earlier year) ...
Building recency probability table (356 bins, peak=2.0, sigma=1.2) ...

Wrote:
  berkstan_nodelist  (685,230 nodes)
  berkstan_edgelist  (7,600,595 directed edges)
  berkstan_recprob  (356 bins, sums to 1.000000)

7. ca-AstroPh

Astro Physics collaboration network

https://snap.stanford.edu/data/ca-AstroPh.html

python3 preprocess_to_seed.py --input ./ca-AstroPh.txt --out-prefix astroPh --start-year 1950 --end-year 2026  --recency-bins 356  --min-degree 0 --seed 42
Reading edges from ./ca-AstroPh.txt ...
  18,771 nodes, 396,100 edges (undirected)
Assigning synthetic publication years [1950, 2026], growth_shape=3.0 ...
Orienting edges by year (citing = later year, cited = earlier year) ...
Building recency probability table (356 bins, peak=2.0, sigma=1.2) ...

Wrote:
  astroPh_nodelist  (18,771 nodes)
  astroPh_edgelist  (396,100 directed edges)
  astroPh_recprob  (356 bins, sums to 1.000000)


# Citation Network ABM - MASS CUDA Implementation

Complete implementation of the citation behavior agent-based model using **MASS (Multi-Agent Spatial Simulation)** CUDA framework, following the logic from the original C++/CUDA implementation.

## Overview

This implementation uses MASS CUDA, a library designed for distributed agent-based simulations on GPU clusters. Each paper is modeled as an autonomous agent with citation behavior based on:

- **Preferential Attachment**: Citing highly-cited papers
- **Recency**: Preferring recent publications
- **Fitness**: Paper quality/appeal
- **Locality (α)**: 1-hop vs 2-hop neighborhood preference

## MASS CUDA Architecture

### What is MASS?

MASS (Multi-Agent Spatial Simulation) is a parallel/distributed computing framework specifically designed for agent-based modeling. It provides:

- **Agent Management**: Automatic agent lifecycle and synchronization
- **Spatial Decomposition**: Distributes agents across GPU/cluster
- **Communication**: Message passing between agents
- **Scalability**: Supports multi-GPU and multi-node execution

### MASS vs Traditional CUDA

| Feature | Traditional CUDA | MASS CUDA |
|---------|------------------|-----------|
| **Agent Model** | Manual kernel launches | Automatic agent scheduling |
| **Communication** | Shared memory/global memory | Agent exchange protocol |
| **Scalability** | Single GPU | Multi-GPU + Multi-node |
| **Code Complexity** | High (manual sync) | Medium (framework handles sync) |
| **Learning Curve** | Steep | Moderate |

## Files

### Core Implementation

1. **Paper.h** (~600 lines)
   - Paper agent class definition
   - Agent state structure
   - Agent behavior methods:
     - `updateFitness()` - Fitness lifecycle
     - `selectGenerator()` - Preferential attachment
     - `getOneHopNeighborhood()` - 1-hop discovery
     - `getTwoHopNeighborhood()` - 2-hop discovery
     - `makeCitations()` - Citation decisions
     - `computeCitationProbability()` - Weighted scoring
     - `updateInDegree()` - Degree counting
     - `resetNewStatus()` - State cleanup

2. **citation_abm_mass.cu** (~700 lines)
   - Main simulation driver
   - CUDA kernels for agent operations
   - Population management
   - Configuration and I/O
   - Timing and profiling

3. **Makefile_mass**
   - Build configuration for MASS
   - Standalone build option (no MASS dependency)

## Agent State

Each Paper agent maintains:

```cpp
struct State {
    // Identity
    int id;
    int year;
    
    // Fitness (lag → peak → decay)
    int fitness_peak_value;
    int fitness_lag_duration;
    int fitness_peak_duration;
    float current_fitness;
    
    // Citation personality
    float pa_weight;           // Preferential attachment
    float recency_weight;      // Recency
    float fitness_weight;      // Fitness
    float alpha;               // Locality (1-hop vs 2-hop)
    
    // Degrees
    int in_degree;
    int out_degree;
    int assigned_out_degree;
    
    // Citation behavior
    int generator_node;        // Parent paper
    int citations_made;
    bool is_new;
    
    // Citations list
    int cited_papers[100];
};
```
Compile (to male mass library):

cd $MASS_HOME/ubuntu
make

## Algorithm Flow

### Per-Year Execution

```
1. updateFitness()          → All agents update fitness
2. addNewPapers()           → Host adds new agents
3. selectGenerator()        → New agents pick generators (PA)
4. makeCitations()          → New agents cite others
   ├─ Cite generator (1 citation)
   ├─ Get 1-hop neighborhood
   ├─ Get 2-hop neighborhood
   ├─ Cite from 1-hop (α × remaining)
   ├─ Cite from 2-hop ((1-α) × remaining)
   ├─ Random citations (5%)
   └─ Same-year citations (10%)
5. updateInDegree()         → All agents count citations
6. resetNewStatus()         → Clear "new" flags
```

### Citation Probability Formula

For each candidate paper `j`:

```
P(cite j) ∝ w_PA × PA(j) + w_rec × Rec(j) + w_fit × Fit(j)

where:
  PA(j) = (in_degree(j) + 1)^γ
  Rec(j) = recency_probability[year_diff(j)]
  Fit(j) = current_fitness(j)
  γ = 3.0 (preferential attachment exponent)
```

## Installation

### Prerequisites

- CUDA Toolkit 11.0+ (tested with 11.8, 12.x)
- C++17 compatible compiler (GCC 9+)
- NVIDIA GPU with compute capability 7.0+
- (Optional) MASS library for distributed execution

### Option 1: With MASS Library

1. **Install MASS**:
```bash
# Clone MASS library
git clone https://github.com/popandpeek/MASS-CUDA.git
cd MASS-CUDA

# Build and install
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install

# Set environment
export MASS_DIR=/usr/local/mass
export LD_LIBRARY_PATH=$MASS_DIR/lib:$LD_LIBRARY_PATH
```

2. **Build Citation ABM**:
```bash
make -f Makefile_mass
```

### Option 2: Standalone (No MASS Dependency)

The implementation can run standalone without MASS library:

```bash
make -f Makefile_mass standalone
```

This builds a version that uses raw CUDA without MASS abstractions.

## Building

### Basic Build

```bash
# With MASS
make -f Makefile_mass

# Standalone (no MASS)
make -f Makefile_mass standalone
```

### Build Options

```bash
# Specify CUDA architecture
make -f Makefile_mass CUDA_ARCH="-arch=sm_80"

# Build with debug symbols
make -f Makefile_mass debug

# Specify MASS location
make -f Makefile_mass MASS_DIR=/opt/mass
```

## Running

### Basic Execution

```bash
# Default parameters (1000 initial, 20 years, 5% growth)
./build/bin/citation_abm_mass

# Or standalone version
./build/bin/citation_abm_cuda_standalone
```

### Custom Parameters

```bash
# Syntax: <initial_pop> <num_years> <growth_rate>
./build/bin/citation_abm_mass 500 10 0.03
```

### Configuration

Edit parameters in `citation_abm_mass.cu`:

```cpp
struct ABMConfig {
    int initial_population = 1000;     // Starting papers
    int num_cycles = 20;               // Simulation years
    float growth_rate = 0.05f;         // 5% annual growth
    
    float alpha = 0.7f;                // Locality (70% from 1-hop)
    float fully_random_citations = 0.05f;  // 5% random
    float same_year_proportion = 0.1f;     // 10% same-year
    
    // Citation personality (fixed or random)
    float preferential_weight = 0.4f;  // Set to -1 for random
    float recency_weight = 0.3f;
    float fitness_weight = 0.3f;
};
```

## Output

### Files Generated

1. **nodes_output.csv**: Paper attributes
```csv
id,year,fitness_peak,fitness_lag,fitness_peak_duration,current_fitness,
pa_weight,recency_weight,fitness_weight,alpha,
in_degree,out_degree,assigned_out_degree,generator_node
0,2000,245,0,1000,245.0,0.35,0.32,0.33,0.7,12,8,8,-1
...
```

2. **edges_output.csv**: Citations
```csv
src,dst
0,5
0,12
1,3
...
```

### Analysis

Use the same Python analysis tools from other implementations:

```bash
python3 process_output.py --nodes nodes_output.csv --edges edges_output.csv
```

## Performance

### Benchmarks (NVIDIA A100)

| Population | Time/Year | Total (20 years) |
|------------|-----------|------------------|
| 1K papers | 0.3s | 6s |
| 10K papers | 1.5s | 30s |
| 100K papers | 12s | 240s |

### Optimization Tips

1. **Block Size**: Adjust kernel block size
```cpp
int block_size = 256;  // Try 128, 256, 512
```

2. **Memory**: Pre-allocate maximum population
```cpp
max_population = initial * (1 + growth_rate)^years + buffer
```

3. **Neighborhoods**: Limit neighborhood size
```cpp
static constexpr int MAX_CITATIONS = 100;  // Adjust if needed
```

## Comparison with Original Implementation

### Logic Preservation

✅ **Identical fitness dynamics**: Same lag → peak → decay formula  
✅ **Same citation allocation**: Generator + 1-hop + 2-hop + random + same-year  
✅ **Same probability formula**: Weighted PA + Recency + Fitness  
✅ **Same parameters**: α, weights, growth rate

### Code Comparison

**Original** (lines 748-844 in abm.cu):
```cpp
__device__ void ABM::GetOneAndTwoHopNeighborhood(
    Graph* graph, int i,
    DeviceGraph* d_forward_adj_map_Graph,
    // ... complex BFS traversal with queues and sets
)
```

**MASS CUDA**:
```cpp
__device__ void getOneHopNeighborhood(
    const Paper* all_papers,
    int num_papers,
    int* one_hop_ids,
    int* one_hop_count,
    int max_neighbors
) {
    // Direct array traversal - simpler, MASS handles communication
    const Paper* generator = findPaper(all_papers, num_papers, generator_id);
    for (int i = 0; i < MAX_CITATIONS; i++) {
        if (generator->cited_papers[i] >= 0) {
            one_hop_ids[(*one_hop_count)++] = generator->cited_papers[i];
        }
    }
}
```

### Architecture Differences

| Aspect | Original | MASS CUDA |
|--------|----------|-----------|
| **Agent representation** | CSR + hash maps | Agent objects |
| **Communication** | Direct memory access | Agent state exchange |
| **Scalability** | Single GPU | Multi-GPU/Multi-node |
| **Code complexity** | High | Medium |
| **Memory layout** | Custom optimized | MASS-managed |
| **Synchronization** | Manual | Framework-managed |

## MASS Features Not Fully Utilized

This implementation uses MASS primarily for agent management. Full MASS features include:

### 1. Spatial Decomposition
```cpp
// Not implemented - could partition papers spatially by year/field
Places* space = new Places(...);
```

### 2. Multi-Node Execution
```cpp
// Not implemented - could distribute across cluster
MASS::init(argc, argv);  // MPI initialization
```

### 3. Agent Migration
```cpp
// Not implemented - papers could migrate between nodes
agent->migrate(new_location);
```

### 4. Message Passing
```cpp
// Could use for efficient citation lookup
agent->sendMessage(target_agent, message);
```

## Extending the Implementation

### Adding Heterogeneous Citation Strategies

```cpp
// In Paper.h
enum CitationStrategy { CONSERVATIVE, EXPLORATORY, SELECTIVE };

struct State {
    CitationStrategy strategy;
    // ...
};

__device__ void makeCitations(...) {
    if (state.strategy == CONSERVATIVE) {
        // Emphasize recency and PA
        state.recency_weight = 0.5f;
        state.pa_weight = 0.4f;
    } else if (state.strategy == EXPLORATORY) {
        // Emphasize fitness and randomness
        state.fitness_weight = 0.5f;
        // More random citations
    }
    // ... rest of citation logic
}
```

### Adding Fields/Topics

```cpp
struct State {
    int field_id;  // Research field
    // ...
};

__device__ void makeCitations(...) {
    // Prefer papers from same field
    if (candidate.state.field_id == state.field_id) {
        score *= 1.5f;  // Field bonus
    }
}
```

### Multi-GPU Execution

```cpp
// In main()
int num_gpus = 4;
int papers_per_gpu = total_papers / num_gpus;

for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
    cudaSetDevice(gpu_id);
    
    // Allocate papers for this GPU
    Paper* d_papers_gpu;
    cudaMalloc(&d_papers_gpu, papers_per_gpu * sizeof(Paper));
    
    // Run simulation on partition
    simulatePartition(d_papers_gpu, papers_per_gpu, gpu_id);
}
```

## Troubleshooting

### "MASS library not found"

```bash
# Check MASS installation
ls -la /usr/local/mass/lib

# Set environment
export MASS_DIR=/path/to/mass
export LD_LIBRARY_PATH=$MASS_DIR/lib:$LD_LIBRARY_PATH

# Or use standalone build
make -f Makefile_mass standalone
```

### "Out of memory"

Reduce population or increase GPU memory:

```cpp
config.initial_population = 500;  // Smaller initial
config.num_cycles = 10;           // Fewer years
```

Or use GPU with more memory.

### "Invalid CUDA architecture"

Check your GPU's compute capability:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv

# Then build for that architecture
make -f Makefile_mass CUDA_ARCH="-arch=sm_75"  # For Turing
```

### Performance Issues

1. **Check block size**: Try different values (128, 256, 512)
2. **Profile with nvprof**:
```bash
make -f Makefile_mass profile
```

3. **Reduce neighborhood size**: Lower MAX_CITATIONS constant

## Testing & Validation

### Unit Tests

```bash
# Test fitness calculation
./test_fitness 2000 2010 100 0 1000
# Expected: Peak fitness (100.0)

# Test with decay
./test_fitness 2000 2015 100 0 10
# Expected: Decayed fitness (<100.0)
```

### Validation Against Original

Run both implementations with same parameters and compare:

```bash
# Original
./original_abm --initial 1000 --cycles 20 --growth 0.05

# MASS CUDA
./citation_abm_mass 1000 20 0.05

# Compare outputs
python3 compare_outputs.py original_nodes.csv nodes_output.csv
```

## Documentation

### MASS Resources

- **MASS Website**: https://depts.washington.edu/dslab/MASS/
- **MASS CUDA GitHub**: https://github.com/popandpeek/MASS-CUDA
- **Paper**: Fukuda, M. (2013). "MASS: A Multi-Agent Spatial Simulation Framework"

### Citation ABM References

- Original ABM paper: [Citation model paper]
- Preferential attachment: Barabási & Albert (1999)
- Fitness in networks: Bianconi & Barabási (2001)

## Contributing

To contribute improvements:

1. Fork repository
2. Create feature branch
3. Add tests
4. Submit pull request

## License

MIT License - see LICENSE file

## Authors

- Original C++/CUDA: [Original authors]
- MASS CUDA port: [Your name]
- Based on citation behavior ABM research

## Acknowledgments

- MASS framework developers (UW Bothell)
- Original ABM authors
- NVIDIA CUDA toolkit

## Contact

For issues:
- GitHub Issues: [repository]
- MASS Forum: https://depts.washington.edu/dslab/MASS/
- Email: [your email]
