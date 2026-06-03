#!/bin/bash
RESDIR=${1:-results/threads}
PREFIX=${2:-n500k}
SUFFIX=${3:-}

T1=$(grep "Time (sec)" "$RESDIR"/${PREFIX}_T1${SUFFIX}_run*.txt 2>/dev/null | \
     awk '{print $NF}' | sort -n | awk 'NR==2')

[ -z "$T1" ] && { echo "ERROR: no T=1 files found in $RESDIR for prefix=$PREFIX suffix=$SUFFIX"; exit 1; }

echo "baseline_time=$T1"
echo "T,time,speedup,efficiency"

for T in 1 2 4 8 16 32; do
    T_MED=$(grep "Time (sec)" "$RESDIR"/${PREFIX}_T${T}${SUFFIX}_run*.txt 2>/dev/null | \
            awk '{print $NF}' | sort -n | awk 'NR==2')
    [ -z "$T_MED" ] && continue
    awk -v t1="$T1" -v tmed="$T_MED" -v threads="$T" \
        'BEGIN { sp=t1/tmed; eff=sp/threads; printf "%d,%.6f,%.3f,%.3f\n", threads, tmed, sp, eff }'
done
