#!/bin/sh

./abm --edgelist sj_edgelist --nodelist sj_nodelist --out-degree-bag tcen_at_least_five --recency-probabilities sj_recprob --same-year-proportion 0.12 --alpha 0.5 --preferential-weight 0.33 --recency-weight 0.33 --fitness-weight 0.33 --growth-rate 0.05 --auxiliary-information-file dummy5-5.aux --num-cycles 5 --output-file res5-5.out --log-file output5-5.log --num-processors 1 --log-level 1
