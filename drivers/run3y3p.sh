#!/bin/sh

./build/bin/citation_abm_mass --edgelist sj_edgelist --nodelist sj_nodelist \
    --out-degree-bag tcen_at_least_five --recency-probabilities sj_recprob \
    --same-year-proportion 0.12 --alpha 0.5 --preferential-weight 0.33 \
    --recency-weight 0.33 --fitness-weight 0.33 --growth-rate 0.03 \
    --auxiliary-information-file output/mass_cuda-static-output-3y-3p.aux \
    --num-cycles 3 --output-file output/mass_cuda-static-output-3y-3p.edgelist \
    --log-file output/mass_cuda-static-output-3y-3p.log \
    --num-processors 1 \
    --log-level 1
