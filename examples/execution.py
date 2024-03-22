import subprocess, os, signal

try:
    logs = subprocess.Popen(["python3","monitoring.py"])
    bgmfast = subprocess.Popen(["taskset", '--cpu-list', '0-12', '/home/malcazar/.local/bin/spark-submit', '--driver-memory', '4g', '--executor-memory', '8g', '--num-executors', '4', '--executor-cores', '2', 'bgmfast_and_abc.py'])
    bgmfast.communicate()
except KeyboardInterrupt:
    os.killpg(os.getpgid(logs.pid), signal.SIGTERM)
    os.killpg(os.getpgid(bgmfast.pid), signal.SIGTERM)

