#!/bin/sh

# Step 1: export (run once per .ncu-rep)
module purge
module load modtree/gpu

ml --force unload xalt

python3 nsys_analyze_full.py  \
 --files profile/nsys-pa-3y-1p.sqlite  "3y/1%" \
 --files profile/nsys-pa-3y-3p.sqlite  "3y/3%" \
 --files profile/nsys-pa-3y-5p.sqlite   "3y/5%" \
 --files profile/nsys-pa-3y-6p.sqlite   "3y/6%" \
 --files profile/nsys-pa-10y-1p.sqlite  "10y/1%" \
 --files profile/nsys-pa-10y-3p.sqlite  "10y/3%" \
 --files profile/nsys-pa-10y-5p.sqlite "10y/5%" \
 --files profile/nsys-pa-10y-6p.sqlite "10y/6%" \
 --files profile/nsys-pa-30y-1p.sqlite  "30y/1%" \
 --files profile/nsys-pa-30y-3p.sqlite  "30y/3%" \
 --files profile/nsys-pa-30y-5p.sqlite  "30y/5%" \
 --files profile/nsys-pa-30y-6p.sqlite    "30y/6%" \
 --gpu H100 --outdir ./pa_H100_paper_output  --prefix pa_H100_vary_years
 
 
