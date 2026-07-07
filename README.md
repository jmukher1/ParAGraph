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


