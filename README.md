The Repository is created to host all the code related to GPU based optimizations for Agent-based Modeling framework for Citation Graph growth.

The master branch is a fork from https://github.com/MinhyukPark/cpp_abm/blob/v5-p which is a c++ implementation (a baseline) to compare against for all the different variations.

There are 6 different branches:

1. Master: c++ implementation (a baseline) to compare against (logically similar to v5-p branch)
2. gpu: gpu implementation similar to (v5-parallel) implementing #1 in gpu
3. gpu-opt: Optimal version of the #2 (optimized gpu version)
4. cpu-model: c++ implementation #1 plus Erdos-Renyi model
5. gpu-model: gpu implementation #3 plus GPU optimized Erdos-Renyi model
6. mass_cuda: MASS_CUDA based mass_cuda based implemenation of Preferential Attachment model (similar to #1 and #2) and Erdos-Renyi model on GPU


