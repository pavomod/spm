#!/bin/bash
RESDIR=${1:-~/gmodica5/results/mpi_omp}

echo "file,np,t,total_time,local_compute,comm,reduction,epoch_transition"

for f in $RESDIR/*.txt; do
    fname=$(basename $f .txt)
    total=$(grep   "^Time (sec)"                $f | awk '{print $NF}')
    local=$(grep   "^local_compute_time_sec"    $f | awk -F= '{print $2}')
    comm=$(grep    "^communication_time_sec"     $f | awk -F= '{print $2}')
    red=$(grep     "^reduction_time_sec"         $f | awk -F= '{print $2}')
    epoch=$(grep   "^epoch_transition_time_sec"  $f | awk -F= '{print $2}')
    np=$(grep      "^num_procs"                  $f | awk -F= '{print $2}' | awk '{print $1}')
    t=$(grep       "^num_threads_per_proc"        $f | awk -F= '{print $2}')
    echo "$fname,$np,$t,$total,$local,$comm,$red,$epoch"
done
