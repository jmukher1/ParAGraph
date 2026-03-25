#!/bin/sh
echo "p = 0.01%"
./drivers/p0001_model_static_vary_p.sh ; 

echo "p = 0.03%"
./drivers/p0003_model_static_vary_p.sh ; 

echo "p = 0.05%"
./drivers/p0005_model_static_vary_p.sh ; 

echo "p = 0.1%"
./drivers/p001_model_static_vary_p.sh ; 

echo "p = 0.5%"
./drivers/p005_model_static_vary_p.sh ; 

echo "p = 1.0%"
./drivers/p01_model_static_vary_p.sh