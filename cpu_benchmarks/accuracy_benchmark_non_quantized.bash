#!/bin/bash
# rm "log1.txt"
COUNTER=0
for f in "benchmark_stories"/*;
do
    cnt=$(cat $f)

    # echo $cnt

    echo $COUNTER
    echo "${cnt:Q}"
    out=$(./run_ppl stories42M.bin -i "${cnt:Q}")
    printf "$out\n" >> "llama_nonquantized_2.txt"

    let COUNTER++
done    
