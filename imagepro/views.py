from django.shortcuts import render
import django_fanout as fanout
import psutil
import time
import os
import re
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
    dim_path = request.GET.get("dim_path","").strip()
    img_counter = 0

    print "option: " + str(option) + " img_path: " + img_path + " dim_path: " + dim_path

    if option == '1' and os.path.isdir(img_path) and img_path != "/":
	filter = request.GET.get("filter","grayscale")
        canny_sigma = request.GET.get("canny_sigma",1)
	print "filter: " + filter + " canny_sigma: " + canny_sigma		
	context = transform(img_path,filter,canny_sigma)		
	return HttpResponse(json.dumps(context), content_type='application/json')

    if option == '2' and os.path.isfile(dim_path) and dim_path != "":
	dim_red = request.GET.get("dim_red","pca").strip()
        dim_k = request.GET.get("dim_k",2)
	print "dim_red: " + dim_red + " K: " + dim_k		
	context = reduce(dim_path,dim_red,dim_k)		
	return HttpResponse(json.dumps(context), content_type='application/json')


#Image Processing
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


#Dim Reduction
    if  dim_path == "":
	dim_info = "please input a valid data file"
        data = {'dim_info': dim_info}
    
    elif not os.path.isfile(dim_path):
	dim_info = "invalid data file"
        data = {'dim_info': dim_info}
	return HttpResponse(json.dumps(data), content_type='application/json')
    
    else:
	n_data = 0
	n_feature = 0
	dim_flag = True
	dim_file = open(dim_path,'r')
        for line in dim_file:
		n_data += 1
		if n_data == 1:
			n_feature = len(line.split(" "))
        	else:
			if n_feature != len(line.split(" ")):
				dim_flag = False
				break
	dim_file.close()
	if dim_flag:		
		dim_info = "Data set " + str(n_data) + " x " + str(n_feature)
	else:
		dim_info = "invalid data file: unbalanced data dimension"
        data = {'dim_info': dim_info}
	print data
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



def reduce(inputpath,alg,k):
	from pyspark import SparkContext
        from pyspark.sql import SQLContext, Row
        from pyspark.mllib.linalg import Vectors
        from pyspark import SparkConf, SparkContext
	n_data = 0
	n_features = 0
	result = "successful!"
	inputdir = os.path.dirname(inputpath)
	print "inputdir: " + inputdir
	inputfile = open(inputpath,'r')
	for line in inputfile:
                input_n = len(line.split(" "))
                print "Selected data set has " + str(input_n) + " features"
                break

        inputfile.close()

	if int(k) >= input_n:
                print "reduced features must be smaller than input features."
                result =  "reduced features must be smaller than input features."
	else:
		os.system("export _JAVA_OPTIONS='-Xms1g -Xmx40g'")
		conf = (SparkConf().set("spark.driver.maxResultSize", "5g"))
                sc = SparkContext(conf=conf)
                sqlContext = SQLContext(sc)
                lines = sc.textFile(inputpath).map(lambda x:x.split(" "))
                lines = lines.map(lambda x:(x[0],[float(y) for y in x[1:]]))
                df = lines.map(lambda x: Row(labels=x[0],features=Vectors.dense(x[1]))).toDF()
	
		if alg == "pca":
			output_data = pca(inputdir,df,alg,k)
			#os.system("spark-submit /home/ubuntu/yi-imPro/imagepro/pca.py " + inputpath + " " + k)

		output_data = inputdir + "/" + alg + str(k) + "_Data"
		inputfile = open(output_data, 'r')
	       	file_size = str(os.stat(output_data).st_size )
        	counter = 0
  	     	n_features = '0'
        	for line in inputfile:
                	input_n = len(line.split(" "))
                	n_features = str(input_n)
                	counter += 1

        	inputfile.close()
        	n_data = str(counter)

                result = "File: " + os.path.basename(output_data) + '</br>'
                result += "Path: " + os.path.dirname(output_data) +  '/' + dim_red + str(k) + "_Features/" + '</br>'
                result += "Dimension: " + n_data + " x " + n_features + "</br>"
                result += "Size: " + file_size + ' bytes'
		print result

        context = {'n_data': n_data, 'n_features': n_features, 'result': result}
	return context


def pca(inputdir,df,alg,k):
        from pyspark.ml.feature import PCA
	pca = PCA(k=int(k),inputCol="features", outputCol="pca_features")
        model = pca.fit(df)
        outData = model.transform(df)
        pcaFeatures = outData.select("labels","pca_features")
	output_data = writeOut(inputdir,pcaFeatures,alg,k)
	return output_data

def writeOut(inputdir,df,alg,k):
	output_dir = inputdir + "/" + alg + str(k) + "_Features"
	output_data = inputdir + "/" + alg + str(k) + "_Data"
	n_data = 0	
	n_features = 0

	df.rdd.repartition(1).saveAsTextFile(output_dir)
        outputfile = open(output_data, 'w')
        inputfile = open(output_dir + '/part-00000', 'r')
        for line in inputfile:
			n_data += 1
                        x = line.split("[")[1].split("]")[0]
                        x = re.sub(',','',x)
                        y = line.split("'")[1]
                        outputfile.write(y + " " + x + '\n')
        inputfile.close()
        outputfile.close()

        print "Dimension reduction finished!"
	return output_data
