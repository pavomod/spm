#!/bin/bash
RESDIR=${1:-results/mpi_omp/strong}
SEQ_LOG=${2:?'Usage: extract_speedup_mpi.sh <strong_resdir> <seq_large.log>'}

[ ! -f "$SEQ_LOG" ] && { echo "ERROR: seq log not found: $SEQ_LOG"; exit 1; }

SEQ_T=$(grep "Time (sec)" "$SEQ_LOG" | awk '{print $NF}')
echo "seq_time=$SEQ_T"
echo "np,nodes,t,total_cores,time_med,speedup,efficiency"

for FILE in "$RESDIR"/strong_n2M_nodes*_np*_t*_run1.txt; do
    [ -f "$FILE" ] || continue
    BASE=$(basename "$FILE" _run1.txt)
    NODES=$(echo "$BASE" | grep -oP '(?<=nodes)\d+')
    NP=$(echo "$BASE"    | grep -oP '(?<=_np)\d+')
    T=$(echo "$BASE"     | grep -oP '(?<=_t)\d+')
    TOTAL=$(( NP * T ))
    MED=$(grep "Time (sec)" "$RESDIR"/${BASE}_run*.txt 2>/dev/null | \
          awk '{print $NF}' | sort -n | awk 'NR==2')
    [ -z "$MED" ] && continue
    awk -v seq="$SEQ_T" -v med="$MED" -v np="$NP" -v nodes="$NODES" -v t="$T" -v total="$TOTAL" \
        'BEGIN { sp=seq/med; eff=sp/total; printf "%d,%d,%d,%d,%.6f,%.3f,%.3f\n", np, nodes, t, total, med, sp, eff }'
done | sort -t, -k1,1n
