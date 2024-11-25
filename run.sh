#!/bin/zsh

for mu in $(seq 0 0.1 1); do
  python MVHSC.py -E 300 --mu $mu --device_set False --file_name "test4mu_cpu"
done