#!/bin/bash
# Basic while loop
counter=1
while [ $counter -le 10 ]
do
    python -m thesis.experiments.att_removal.attn_search "car" "parking lot" attnsearch/$counter --seed $counter
    cp -r attnsearch/$counter /share/projects/demystify/attnmaps/
    ((counter++))
done
echo All done