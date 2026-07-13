#!/bin/sh

# Step 1: export (run once per .ncu-rep)
module purge
module load modtree/gpu

ml --force unload xalt

python3 nsys_occupancy_multigpu.py \
  --gpu H100 \
  --files profile/nsys-pa-30y-1p.sqlite  "30y/1%" \
 --files profile/nsys-pa-30y-3p.sqlite  "30y/3%" \
 --files profile/nsys-pa-30y-6p.sqlite    "30y/6%" \
  --outdir ./nsys_output --prefix pa_30y

 
