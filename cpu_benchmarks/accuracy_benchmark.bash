#!/bin/bash
rm "log1.txt"
COUNTER=0
for f in "benchmark_stories"/*;
do
    cnt=$(cat $f)

    # echo $cnt

    echo $COUNTER
    echo "${cnt:Q}"
    out=$(./runq_ppl modelq.bin -i "${cnt:Q}")
    printf "$out\n" >> "llama_quantized_2.txt"

    let COUNTER++
done    
