from django.shortcuts import render
import django_fanout as fanout
import psutil
import time
import os
#from django.template.context_processors import csrf
import json
from django.http import HttpResponse

#log_file = "/home/ubuntu/yi-imPro/imagepro/static/imagepro/logfile.txt"
#fanout.publish('test', 'Test publish!')

def index(request):
    option = request.GET.get("option",0)
    print "it is " + str(option)
    img_path = request.GET.get("img_path","/...").strip()
    img_counter = 0
    if  img_path == "" or img_path == "/...":
	img_info = "please input a valid image directory"
        data = {'img_info': img_info}
    elif not os.path.isdir(img_path):
	img_info = "invalid directory"
        data = {'img_info': img_info}
	return HttpResponse(json.dumps(data), content_type='application/json')
    else:
        for file in img_path:
		if ".png" or ".jpg" in file:
			img_counter += 1
        img_info = str(img_counter) + " images found"
        data = {'img_info': img_info}
	return HttpResponse(json.dumps(data), content_type='application/json')
    
    context = {'img_path': img_path, 'img_info': img_info}

    return render(request, 'imagepro/index.html', context)

def terminal(request):
    #logfile = open(log_file,'r').read()
    logfile = "imPro v1.0.1 \n Terminal loading  . . ."
    context = {'log': logfile, 'test': 'hello <br/> world'}
    return render(request, 'imagepro/terminal.html', context)
