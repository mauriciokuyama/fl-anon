#!/usr/bin/env bash

mkdir -p "/tmp/.flwr/$2"
mkdir -p "results/$2/$1"

input_file="/tmp/.flwr/$2/$1.log"
output_file="results/$2/$1/metrics_accuracy.csv"
output_file2="results/$2/$1/metrics_loss.csv"

echo flwr run . --run-config "k=$1 anon-method='$2'"
flwr run . --run-config "k=$1 anon-method='$2'" &> "$input_file"


#cut -c 23- "$input_file" | grep "test_accuracy" | sed "s/.*'test_accuracy': //; s/],.*//; s/[()]/ /g; s/,/ /g" | awk '{for(i=1;i<=NF;i+=2) print $i","$((i+1))}' > "$output_file"
#cut -c 23- "$input_file" | awk '/test_accuracy/{flag=1; next} flag && /\]/{flag=0} flag' | tr -d '()' | tr ',' ' ' | awk '{for(i=1;i<=NF;i+=2) print $i","$((i+1))}' > "$output_file"

# Create the output file and add the header
echo "round,accuracy" > "$output_file"  # Write header to CSV
echo "round,loss" > "$output_file2"  # Write header to CSV

# Extract test_accuracy values
cut -c 23- "$input_file" | \
awk "/'test_accuracy'/{flag=1; print; next} flag {print} flag && /\]/{flag=0}" | \
sed "s/'test_accuracy': //" | \
tr -d '()[]{}' | \
tr ',' ' ' | \
awk '{for(i=1;i<=NF;i+=2) print $i","$((i+1))}' >> "$output_file"

# Extract loss values
out=$(cut -c 24- "$input_file" | grep 'round [1-9]')
out=${out//'round '/''}
out=${out//': '/','}
echo "$out" >> "$output_file2"

echo "Extracted test_accuracy values to $output_file"
echo "Extracted loss values to $output_file2"
