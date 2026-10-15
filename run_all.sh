#!/usr/bin/env bash
# Reproduce the whole project: data -> experiments -> figures.  Total runtime is about 5 minutes on a laptop CPU.
set -e
python download_data.py
python 01_dataset_stats.py        | tee results/log_01_dataset_stats.txt
python 02_keyphrase_experiments.py | tee results/log_02_keyphrase_experiments.txt
python 03_topic_organization.py    | tee results/log_03_topic_organization.txt
python 04_error_analysis.py        | tee results/log_04_error_analysis.txt
python 05_make_figures.py
echo "All done. See the results/ folder."
