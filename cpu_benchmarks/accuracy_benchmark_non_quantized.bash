#!/bin/bash
# rm "log1.txt"
COUNTER=0
for f in "benchmark_stories"/*;
do
    cnt=$(cat $f)

    # echo $cnt

    echo $COUNTER
    echo "${cnt:Q}"
    out=$(./run_ppl stories110M.bin -i "${cnt:Q}")
    printf "$out\n" >> "llama_nonquantized.txt"

    let COUNTER++
done    
