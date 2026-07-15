The Repository is created to host all the code related to ParAGraph: A GPU based optimizations for Agent-based Modeling framework for Graph growth.


There are 5 different branches:

1. Master: Optimal version of the ParAGraph (optimized gpu version)
2. cpu-model: c++ implementation of preferential attachment plus Erdos-Renyi model
3. gpu-er: gpu implementation GPU optimized Erdos-Renyi model
4. masscuda: MASS_CUDA based mass_cuda based implemenation of Preferential Attachment model 
5. masscuda-er: MASS_CUDA based mass_cuda based implemenation of Erdos-Renyi model on GPU and #4 

Pre-requisite:
mass_cuda_core
cuCollections (NVIDIA)
NVIDIA Toolkit
