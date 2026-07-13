
Store important metrics in csv:

for i in `ls profile/ncu*.ncu-rep`; do echo $i; ncu --import $i --csv > "${i%.ncu-rep}.csv" ; done

Print Summary

for i in `ls profile/ncu*.ncu-rep`; do echo $i; ncu --import $i --print-summary per-kernel  > "${i%.ncu-rep}-kernel-summary.txt" ; done
