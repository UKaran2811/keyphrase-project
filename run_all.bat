@echo off
REM Windows version of run_all.sh
python download_data.py || exit /b 1
python 01_dataset_stats.py > results\log_01_dataset_stats.txt || exit /b 1
python 02_keyphrase_experiments.py > results\log_02_keyphrase_experiments.txt || exit /b 1
python 03_topic_organization.py > results\log_03_topic_organization.txt || exit /b 1
python 04_error_analysis.py > results\log_04_error_analysis.txt || exit /b 1
python 05_make_figures.py || exit /b 1
echo All done. See the results folder.
