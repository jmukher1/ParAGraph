#!/bin/sh

./abm --edgelist sj_edgelist --nodelist sj_nodelist --out-degree-bag tcen_at_least_five --recency-probabilities sj_recprob --same-year-proportion 0.12 --alpha 0.5 --preferential-weight 0.33 --recency-weight 0.33 --fitness-weight 0.33 --growth-rate 0.06 --auxiliary-information-file dummy30-6.aux --num-cycles 30 --output-file res30-6.out --log-file output30-6.log --num-processors 1 --log-level 1
