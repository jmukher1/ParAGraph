#!/bin/sh

# Step 1: export (run once per .ncu-rep)
module purge
module load modtree/gpu

ml --force unload xalt

python3 nsys_export_csv.py  \
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
 --gpu P100 --outdir ./csv_P100  --prefix pa


python3 nsys_plot_figures.py \
    --occupancy ./csv_P100/pa_occupancy.csv \
    --gpu-util  ./csv_P100/pa_gpu_util.csv \
    --memcpy    ./csv_P100/pa_memcpy.csv \
    --outdir    ./csv_P100 --prefix pa
