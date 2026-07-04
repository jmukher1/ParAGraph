The Repository is created to host all the code related to GPU based optimizations for Agent-based Modeling framework for Citation Graph growth.

The master branch is a fork from https://github.com/MinhyukPark/cpp_abm/blob/v5-p which is a c++ implementation (a baseline) to compare against for all the different variations.

There are 6 different branches:

1. Master: c++ implementation (a baseline) to compare against (logically similar to v5-p branch)
2. gpu: gpu implementation similar to (v5-parallel) implementing #1 in gpu
3. gpu-opt: Optimal version of the #2 (optimized gpu version)
4. cpu-model: c++ implementation #1 plus Erdos-Renyi model
5. gpu-model: gpu implementation #3 plus GPU optimized Erdos-Renyi model
6. mass_cuda: MASS_CUDA based mass_cuda based implemenation of Preferential Attachment model (similar to #1 and #2) and Erdos-Renyi model on GPU


com-Amazon

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


https://snap.stanford.edu/data/com-Amazon.html
com-Amazon

python3 amazon_to_citation_seed.py --input ./com-amazon.ungraph.txt --out-prefix amz --start-year 1950 --end-year 2020  --recency-bins 356  --min-degree 0 --seed 4i2

Reading edges from ./com-amazon.ungraph.txt ...
  334,863 nodes, 925,872 edges (undirected)
Assigning synthetic publication years [1950, 2020], growth_shape=3.0 ...
Orienting edges by year (citing = later year, cited = earlier year) ...
Building recency probability table (356 bins, peak=2.0, sigma=1.2) ...

Wrote:
  amz_nodelist  (334,863 nodes)
  amz_edgelist  (925,872 directed edges)
  amz_recprob  (356 bins, sums to 1.000000)

amz_edgelist  amz_nodelist  amz_recprob


