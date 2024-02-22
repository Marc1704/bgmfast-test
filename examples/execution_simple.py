import subprocess

bgmfast = subprocess.Popen(["taskset", '--cpu-list', '0-12', '/home/hpc/antares/malcazar/.local/bin/spark-submit', '--driver-memory', '4g', '--executor-memory', '8g', '--num-executors', '4', '--executor-cores', '2', 'bgmfast_redundant_test.py'])

