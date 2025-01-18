#!/bin/bash
python baseline_with_repetitions.py --start-k 2 --stop-k 100 "$1" knn "$2" &
python baseline_with_repetitions.py --start-k 2 --stop-k 100 "$1" xgb "$2" &
python baseline_with_repetitions.py --start-k 2 --stop-k 100 "$1" rf "$2" &
python baseline_with_repetitions.py --start-k 2 --stop-k 100 "$1" svm "$2"
