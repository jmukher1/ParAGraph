#!/bin/sh


python3 nsys_analyze_full.py  \
 --files erprofile/nsys-er-gnp-3y-1p.sqlite  "3y/1%" \
 --files erprofile/nsys-er-gnp-3y-3p.sqlite  "3y/3%" \
 --files erprofile/nsys-er-gnp-3y-5p.sqlite   "3y/5%" \
 --files erprofile/nsys-er-gnp-3y-6p.sqlite   "3y/6%" \
 --files erprofile/nsys-er-gnp-10y-1p.sqlite  "10y/1%" \
 --files erprofile/nsys-er-gnp-10y-3p.sqlite  "10y/3%" \
 --files erprofile/nsys-er-gnp-10y-5p.sqlite "10y/5%" \
 --files erprofile/nsys-er-gnp-10y-6p.sqlite "10y/6%" \
 --files erprofile/nsys-er-gnp-30y-1p.sqlite  "30y/1%" \
 --files erprofile/nsys-er-gnp-30y-3p.sqlite  "30y/3%" \
 --files erprofile/nsys-er-gnp-30y-5p.sqlite  "30y/5%" \
 --files erprofile/nsys-er-gnp-30y-6p.sqlite    "30y/6%" \
 --gpu P100 --outdir ./er-gnp_P100_paper_output  --prefix er-gnp_P100_vary_years
