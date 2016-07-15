from django.shortcuts import render
import django_fanout as fanout
import psutil
import time
import os
#from django.template.context_processors import csrf
import json
from django.http import HttpResponse
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from PIL import Image
from skimage import filters
from skimage.color import rgb2gray
from skimage import feature
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
from scipy import ndimage as ndi
import math
from skimage.morphology import skeletonize

def as_gray(image_filter, image, *args, **kwargs):
    gray_image = rgb2gray(image)
    return image_filter(gray_image, *args, **kwargs)

@adapt_rgb(as_gray)
def original_gray(image):
    return image

@adapt_rgb(as_gray)
def skeleton_gray(image):
    return skeletonize(image)

@adapt_rgb(as_gray)
def canny_gray(image,p):
    return feature.canny(image,sigma=p)

@adapt_rgb(as_gray)
def sobel_gray(image):
    return filters.sobel(image)

@adapt_rgb(as_gray)
def roberts_gray(image):
    return filters.roberts(image)

def index(request):
    result = ""
    option = request.GET.get("option",0)
    img_path = request.GET.get("img_path","").strip()
    canny_sigma = request.GET.get("canny_sigma",1)
    img_counter = 0
    print "option: " + str(option) + " path: " + img_path + " sigma: " + str(canny_sigma)

    if option == '1' and os.path.isdir(img_path) and img_path != "/":
	filter = request.GET.get("filter","grayscale")
	print "filter: " + filter	
	
	context = transform(img_path,filter,canny_sigma)		
	return HttpResponse(json.dumps(context), content_type='application/json')
#        return render(request, 'imagepro/index.html', context)

    if  img_path == "":
	img_info = "please input a valid image directory"
        data = {'img_info': img_info}
    
    elif not os.path.isdir(img_path):
	img_info = "invalid directory"
        data = {'img_info': img_info}
	return HttpResponse(json.dumps(data), content_type='application/json')
    
    else:
        for file in os.listdir(img_path):
		if file.lower().endswith(('.png','.jpg','.jpeg','.gif')):
			img_counter += 1
        img_info = str(img_counter) + " images found"
        data = {'img_info': img_info}
	return HttpResponse(json.dumps(data), content_type='application/json')
    
	

    context = {'result': result}
    return render(request, 'imagepro/index.html', context)

def terminal(request):
    logfile = "imPro v1.0.1 \n Terminal loading  . . ."
    context = {'log': logfile, 'test': 'hello <br/> world'}
    return render(request, 'imagepro/terminal.html', context)


def transform(inputpath,filter,canny_sigma):
        
        T_size = (80,20)
        L_size = (35,65)
        R_size = (35,65)

        T_box = (1, 1, 1400, 350)
        L_box = (1, 350, 350, 1000)
        R_box = (1050, 350, 1400, 1000)

        counter = 0
        max_count = 2
	if not os.path.exists(inputpath+'/impro_out/'):
    	  os.makedirs(inputpath+'/impro_out/')

	n_files = 0
	for file in os.listdir(inputpath):
                if file.lower().endswith(('.png','.jpg','.jpeg','.gif')):
                        n_files += 1

	i = 0
	if filter == 'canny':
	        cutfile = open(inputpath + '/impro_out/' + filter + canny_sigma + '_processed_data', 'w')
	else:
	        cutfile = open(inputpath + '/impro_out/' + filter + '_processed_data', 'w')

        for file in os.listdir(inputpath):
	  if file.lower().endswith(('.png','.jpg','.jpeg','.gif')):
            
	    try:
		i += 1    
		print("compressing..." + file + ' ' +  str(i) + '/' + str(n_files))
                im = Image.open(inputpath + "/" + file)
                cutfile.write(file.split(".")[0])
	######Filters####
		if filter == 'grayscale':
			im = original_gray(np.array(im))
			im = Image.fromarray(np.uint8(im*255))
		if filter == 'sobel':
			im = 1-sobel_gray(np.array(im))
			im = Image.fromarray(np.uint8(im*255))
		if filter == 'roberts':
			im = 1-roberts_gray(np.array(im))
			im = Image.fromarray(np.uint8(im*255))
		if filter == 'canny':
			im = 1-canny_gray(np.array(im),canny_sigma)
			im = Image.fromarray(np.uint8(im*255))
		if filter == 'skeleton':
			im = 1-skeleton_gray(np.array(im))
			im = Image.fromarray(np.uint8(im*255))


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
	
	if filter == 'canny':
		path = inputpath +  '/impro_out/' + filter + canny_sigma + '_processed_data'
	else:
		path = inputpath +  '/impro_out/' + filter + '_processed_data'
	inputfile = open(path, 'r')
	file_size = str(os.stat(path).st_size )
	counter = 0
	n_features = '0'	
	for line in inputfile:
                input_n = len(line.split(" "))
                n_features = str(input_n)
		counter += 1
	#	break

        inputfile.close()
        n_data = str(counter)
	if filter == 'canny':
		result = "File: " + filter + canny_sigma + '_processed_data </br>'
		result += "Path: " + inputpath +  '/impro_out/ </br>'
		result += "Dimension: " + n_data + " x " + n_features + "</br>"
		result += "Size: " + file_size + ' bytes' 
	else:	
		result = "File: " + filter + "_processed_data </br>"
		result += "Path: " + inputpath +  '/impro_out/ </br>'
		result += "Dimension: " + n_data + " x " + n_features + "</br>"
		result += "Size: " + file_size + ' bytes' 
	print result

        context = {'n_data': n_data, 'n_features': n_features, 'result': result}
	return context
