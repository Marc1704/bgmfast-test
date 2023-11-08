#!/usr/bin/env python

import os, platform, subprocess, socket, psutil, netifaces, cpuinfo, time
from datetime import datetime

kb = float(1024)
mb = float(kb ** 2)
gb = float(kb ** 3)

memTotal = int(psutil.virtual_memory()[0]/gb)
memFree = int(psutil.virtual_memory()[1]/gb)
storageTotal = int(psutil.disk_usage('/')[0]/gb)
info = cpuinfo.get_cpu_info()['brand_raw']

datetime_now = str(datetime.now().strftime("%Y_%m_%dT%H_%M_%S"))
output_file = datetime_now + '_bgmfast_comp.log'
with open(output_file, 'w') as logsfile:
    logsfile = open(output_file, 'w')
    core = os.cpu_count()
    host = socket.gethostname()
    logsfile.write('#System Info\n')
    logsfile.write("#Hostname     : " + str(host) + '\n')
    logsfile.write("#System       : " + str(platform.system()) + ' ' + str(platform.machine()) + '\n')
    logsfile.write("#Kernel       : " + str(platform.release()) + '\n')
    logsfile.write('#Compiler     : ' + str(platform.python_compiler()) + '\n')
    logsfile.write('#CPU          : ' + str(info) + ' ' + str(core) + " (Core)" + '\n')
    logsfile.write("#Memory       : " + str(memTotal) + " GiB" + '\n')
    logsfile.write("#Disk         : " + str(storageTotal) + " GiB" + '\n')
    logsfile.write('#---------------------------------\n')
    logsfile.write('#Units\n')
    logsfile.write('#[running_processes] = process\n')
    logsfile.write('#[load_avg_1_min] = CPU\n')
    logsfile.write('#[load_avg_5_min] = CPU\n')
    logsfile.write('#[load_avg_15_min] = CPU\n')
    logsfile.write('#[ram_used] = GiB\n')
    logsfile.write('#[ram_used_percentatge] = \%\n')
    logsfile.write('#[disk_used] = GiB\n')
    logsfile.write('#[disk_used_percentatge] = \%\n')
    logsfile.write('#[active_interface] = None\n')
    logsfile.write('#[packet_send] = KiB/s\n')
    logsfile.write('#[packet_receive] = KiB/s\n')
    logsfile.write('#---------------------------------\n')
    logsfile.write('datetime,running_processes,load_avg_1_min,load_avg_5_min,load_avg_15_min,ram_used,ram_used_percent,disk_used,disk_used_percent,active_interface,packet_send,packet_receive\n')

while True:
    datetime_now = str(datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
    pidTotal = len(psutil.pids())
    load_avgs = [round(loadavg, 2) for loadavg in os.getloadavg()]
    memUsed = round(psutil.virtual_memory()[3]/gb, 2)
    memPercent = round(memUsed/memTotal*100, 2)
    storageUsed = round(psutil.disk_usage('/')[1]/gb, 2)
    storagePercent = round(storageUsed/storageTotal*100, 2)
    active = netifaces.gateways()['default'][netifaces.AF_INET][1]
    speed = psutil.net_io_counters(pernic=False)
    sent = speed[0]
    psend = round(speed[2]/kb, 2)
    precv = round(speed[3]/kb, 2)
    with open(output_file, 'a') as logsfile:
        logsfile.write(datetime_now + ',' + str(pidTotal) + ',' + str(load_avgs[0]) + ',' + str(load_avgs[1]) + ',' + str(load_avgs[2]) + ',' + str(memUsed) + ',' + str(memPercent) + ',' + str(storageUsed) + ',' + str(storagePercent) + ',' + str(active) + ',' + str(psend) + ',' + str(precv) + '\n')
    time.sleep(5)
