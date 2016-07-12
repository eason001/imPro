from django.shortcuts import render
import django_fanout as fanout
import psutil
import time

log_file = "/home/ubuntu/yi-imPro/imagepro/static/imagepro/logfile.txt"
bg_proc = "/home/ubuntu/yi-imPro/imagepro/static/imagepro/bg_proc.py"

#fanout.publish('test', 'Test publish!')

def index(request):
    context = {'hello': 'world'}
    for i in range(100):
      fanout.publish('test', str(i))
#    while True:
#        fanout.publish('cpupercent', 'CPU percent: {0}%'.format(psutil.cpu_percent()))
    return render(request, 'imagepro/index.html', context)


def terminal(request):
    logfile = open(log_file,'r').read()
    context = {'log': logfile, 'test': 'hello <br/> world'}
    return render(request, 'imagepro/terminal.html', context)
