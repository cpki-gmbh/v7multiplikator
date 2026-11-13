@echo off
rem measure_winner_genomes.bat -- benchmark archived winner genomes.
rem Usage: measure_winner_genomes.bat [--mb 16] [--seconds 2] [--stride 10]
rem Default: every 10th genome (plus genome 1), 16 MB buffer, 2 s per genome.
setlocal
cd /d %~dp0..\
python measure_winner_genomes.py %*
exit /b %errorlevel%