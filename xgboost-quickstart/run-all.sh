#!/usr/bin/env bash

methods=(cb mondrian tdg gfkmc)
k=(3 5 10 20)
# for i in "${k[@]}"; do ./run.sh "$i" mondrian; done
for m in "${methods[@]}"; do
    for i in "${k[@]}"; do
        ./run.sh "$i" "$m" &
    done
    wait
done
