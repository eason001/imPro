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
    result = ""
    option = request.GET.get("option",0)
    print "option: " + str(option)
    img_path = request.GET.get("img_path","").strip()
    img_counter = 0

    if option == '1' and os.path.isdir(img_path):
	filter = request.GET.get("filter","grayscale")
	print "filter: " + filter	
	if filter == 'grayscale':
	        context = grayscale(img_path)
		
        return render(request, 'imagepro/index.html', context)


    if  img_path == "":
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
    
	

    context = {'result': result}
    return render(request, 'imagepro/index.html', context)

def terminal(request):
    #logfile = open(log_file,'r').read()
    logfile = "imPro v1.0.1 \n Terminal loading  . . ."
    context = {'log': logfile, 'test': 'hello <br/> world'}
    return render(request, 'imagepro/terminal.html', context)


def grayscale(inputpath):
	from PIL import Image

        T_size = (80,20)
        L_size = (35,65)
        R_size = (35,65)

        T_box = (1, 1, 1400, 350)
        L_box = (1, 350, 350, 1000)
        R_box = (1050, 350, 1400, 1000)

        counter = 0
        max_count = 2

	cutfile = open(inputpath + '/processed_data', 'w')

        for file in os.listdir(inputpath):
          if '.png' or '.jpg' in file:
            try:    
		print("compressing..." + file)
                im = Image.open(inputpath + "/" + file)
                cutfile.write(file.split(".")[0])

        ######TOP REGION######
                region = im.crop(T_box)
                region.thumbnail(T_size)
                region = region.convert('LA')
                imarray = list(region.getdata())
                for item in imarray:
                        counter = 0
                        for x in item:
                                counter += 1
                                if counter < max_count:
                                        cutfile.write(" " + str(x))
	######LEFT REGION######
                region = im.crop(L_box)
                region.thumbnail(L_size)
                region = region.convert('LA')
                imarray = list(region.getdata())
                for items in imarray:
                        counter = 0
                        for x in item:
                                counter += 1
                                if counter < max_count:
                                        cutfile.write(" " + str(x))

        ######RIGHT REGION######
                region = im.crop(R_box)
                region.thumbnail(R_size)
                region = region.convert('LA')
                imarray = list(region.getdata())
                for items in imarray:
                        counter = 0
                        for x in item:
                                counter += 1
                                if counter < max_count:
                                        cutfile.write(" " + str(x))

                cutfile.write('\n')

            except Exception,e:
                print file + " failed: " + str(e)
		continue
    
        cutfile.close()

	inputfile = open(inputpath + '/processed_data', 'r')
	counter = 0
	for line in inputfile:
                input_n = len(line.split(" "))
                n_features = str(input_n)
		counter += 1
	#	break

        inputfile.close()
        n_data = str(counter)
	print "processed data set: " + n_data + " x " + n_features
	result = "processed data set: " + n_data + " x " + n_features + " is saved as " + inputpath + "/processed_data"

        context = {'n_data': n_data, 'n_features': n_features, 'result': result}
	return context
