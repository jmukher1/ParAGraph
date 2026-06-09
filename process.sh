#!/bin/sh

# Step 1: export (run once per .ncu-rep)
module purge
module load modtree/gpu

ml --force unload xalt

for f in profile/ncu-pa-*.ncu-rep; do
    ncu --import $f --csv --page details > profile/$(basename $f .ncu-rep)-details.csv
done

