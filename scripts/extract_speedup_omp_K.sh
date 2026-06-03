#!/bin/bash
RESDIR=${1:-results/omp_tasks}
T=${2:-16}

echo "K,time_med"
for K in 16 64 256 1024 4096; do
    MED=$(grep "Time (sec)" $RESDIR/n500k_T${T}_K${K}_run*.txt 2>/dev/null | \
          awk '{print $NF}' | sort -n | awk 'NR==2')
    [ -n "$MED" ] && echo "$K,$MED"
done
