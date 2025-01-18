#!/usr/bin/env bash

mkdir -p "/tmp/.flwr/$2"
mkdir -p "results/$2/$1"

input_file="/tmp/.flwr/$2/$1.log"
output_file="results/$2/$1/metrics_auc.csv"
output_file2="results/$2/$1/metrics_acc.csv"

echo flwr run . --run-config "k=$1 anon-method='$2'"
flwr run . --run-config "k=$1 anon-method='$2'" &> "$input_file"


echo "round,AUC" > "$output_file"  # Write header to CSV

cut -c 23- "$input_file" | \
awk "/'AUC'/{flag=1; print; next} flag {print} flag && /\]/{flag=0}" | \
sed "s/'AUC': //" | \
tr -d '()[]{}' | \
tr ',' ' ' | \
awk '{for(i=1;i<=NF;i+=2) print $i","$((i+1))}' >> "$output_file"

echo "Extracted AUC values to $output_file"



echo "round,accuracy" > "$output_file2"  # Write header to CSV

cut -c 23- "$input_file" | \
awk "/'test_accuracy'/{flag=1; print; next} flag {print} flag && /\]/{flag=0}" | \
sed "s/'test_accuracy': //" | \
tr -d '()[]{}' | \
tr ',' ' ' | \
awk '{for(i=1;i<=NF;i+=2) print $i","$((i+1))}' >> "$output_file2"

echo "Extracted ACC values to $output_file2"


#INFO :          {'AUC': [(1, 0.7599945091514143),
#INFO :                   (2, 0.8029174515235458),
#INFO :                   (3, 0.5480249307479224),
#INFO :                   (4, 0.6613487562189054),
#INFO :                   (5, 0.8142926397343665),
#INFO :                   (6, 0.8101538802660755),
#INFO :                   (7, 0.8183454344216934),
#INFO :                   (8, 0.8001079295154185),
#INFO :                   (9, 0.7454985619469027),
#INFO :                   (10, 0.7992545454545454)],
#INFO :           'test_accuracy': [(1, 0.7520798668885191),
#INFO :                             (2, 0.7506925207756233),
#INFO :                             (3, 0.7423822714681441),
#INFO :                             (4, 0.7435046987285793),
#INFO :                             (5, 0.7775318206972883),
#INFO :                             (6, 0.7577605321507761),
#INFO :                             (7, 0.7830658550083011),
#INFO :                             (8, 0.7797356828193832),
#INFO :                             (9, 0.7881637168141593),
#INFO :                             (10, 0.7710643015521065)]}
