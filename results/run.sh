#!/bin/bash

python experiments/correction.py data/ --codebook-size 8 | tee results/correction_3b.csv
python experiments/correction.py data/ --codebook-size 4 | tee results/correction_2b.csv
python experiments/correction.py data/ --codebook-size 3 | tee results/correction_1.5b.csv
python experiments/correction.py data/ --codebook-size 2 | tee results/correction_1b.csv

python experiments/ordering.py data/ --codebook-size 8 | tee results/ordering_3b.csv
python experiments/ordering.py data/ --codebook-size 4 | tee results/ordering_2b.csv
python experiments/ordering.py data/ --codebook-size 3 | tee results/ordering_1.5b.csv
python experiments/ordering.py data/ --codebook-size 2 | tee results/ordering_1b.csv

python experiments/scaling.py data/ --codebook-size 8 --run-obq --run-hessian | tee results/scaling_3b.csv
python experiments/scaling.py data/ --codebook-size 4 --run-obq --run-hessian | tee results/scaling_2b.csv
python experiments/scaling.py data/ --codebook-size 3 --run-obq --run-hessian | tee results/scaling_1.5b.csv
python experiments/scaling.py data/ --codebook-size 2 --run-obq --run-hessian | tee results/scaling_1b.csv

python experiments/local_search.py data/ --codebook-size 8 | tee results/local_search_3b.csv
python experiments/local_search.py data/ --codebook-size 4 | tee results/local_search_2b.csv
python experiments/local_search.py data/ --codebook-size 3 | tee results/local_search_1.5b.csv
python experiments/local_search.py data/ --codebook-size 2 | tee results/local_search_1b.csv

python experiments/dampening.py data/ --codebook-size 8 | tee results/dampening_3b.csv
python experiments/dampening.py data/ --codebook-size 4 | tee results/dampening_2b.csv
python experiments/dampening.py data/ --codebook-size 3 | tee results/dampening_1.5b.csv
python experiments/dampening.py data/ --codebook-size 2 | tee results/dampening_1b.csv

python experiments/compare.py data/ --codebook-size 8 | tee results/compare_3b.csv
python experiments/compare.py data/ --codebook-size 4 | tee results/compare_2b.csv
python experiments/compare.py data/ --codebook-size 3 | tee results/compare_1.5b.csv
python experiments/compare.py data/ --codebook-size 2 | tee results/compare_1b.csv

python experiments/bits.py data/ | tee results/bits.csv
