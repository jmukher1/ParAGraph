The Repository is created to host all the code related to ParAGraph: A GPU based optimizations for Agent-based Modeling framework for Graph growth.


There are 5 different branches:

1. Master: Optimal version of the ParAGraph (optimized gpu version)
2. cpu-model: c++ implementation of preferential attachment plus Erdos-Renyi model
3. gpu-er: gpu implementation GPU optimized Erdos-Renyi model
4. masscuda: MASS_CUDA based mass_cuda based implemenation of Preferential Attachment model 
5. masscuda-er: MASS_CUDA based mass_cuda based implemenation of Erdos-Renyi model on GPU and #4 


## Models Implemented

### 1. Preferential Attachment (PA) - Default
The PA model simulates realistic citation behavior based on:
- **Preferential Attachment**: Popular papers get more citations (rich-get-richer)
- **Recency**: Recent papers are more likely to be cited
- **Fitness**: Papers have intrinsic quality affecting citation probability
- **Neighborhood**: Citations favor papers in local network neighborhoods 

### 3. Erdős-Rényi G(n,p) (ER-GNP)
Each new node connects to each existing node independently with probability p.
- Classical random graph model
- Variable out-degree (binomial distribution)
- Stochastic edge formation

## Directory Structure

```
ParAGraph/
├── src/
│   ├── All .cu/cpp files          # source code
├── gpu-std/
│   ├── Header files          # GPU-Native std framework
├── includes/
│   ├── ParAGraph specific header files        # Command-line parsing
├── build/                # Compiled object files
├── drivers/             # Example run scripts
├── sbatch_scripts/      # Example sbatch scripts to submit the drivers (above)
├── profiling_scripts/   # Example sbatch scripts to run profiling using ncu and nsys
├── makefile              # Build configuration
└── README.md             # This file
```

## Installation

### Prerequisites
- mass_cuda_core
- cuCollections (NVIDIA)
- NVIDIA Toolkit 
- OpenMP for parallelization
- Make

### Build
```bash
make clean
make
```

This creates the `abm` executable.

## Usage

### Basic Command Structure
 
 
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
