from django.shortcuts import render
from multiprocessing.pool import ThreadPool
import django_fanout as fanout
import subprocess
import psutil
import time
import os
import re
import threading
#from django.template.context_processors import csrf
import json
from django.http import HttpResponse
from django.http import StreamingHttpResponse
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
from django.views.decorators.http import condition


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



@condition(etag_func=None)
def index(request):
    result = ""
    option = request.GET.get("option",0)
    img_path = request.GET.get("img_path","").strip()
    dim_path = request.GET.get("dim_path","").strip()
    clu_path = request.GET.get("clu_path","").strip()
    img_counter = 0

    print "option: " + str(option) + " img_path: " + img_path + " dim_path: " + dim_path + " clu_path: " + clu_path

    if option == '1' and os.path.isdir(img_path) and img_path != "/":
	filter = request.GET.get("filter","grayscale")
        canny_sigma = request.GET.get("canny_sigma",1)
	print "filter: " + filter + " canny_sigma: " + canny_sigma		

#	t = ThreadPool(processes=2)
#	t_result = t.apply_async(transform, (img_path,filter,canny_sigma))
#	t_val = t_result.get()
#	print t_val['result']

	t = threading.Thread(target=transform, args=(img_path,filter,canny_sigma), name='transform')
	t.start()
	
	result = "Job Submitted! </br>"
	result += "Description: Processing " + img_path + " with " + filter + "</br>"

	if filter == 'canny':
		result += "File: " + filter + canny_sigma + '_processed_data </br>'
		result += "Path: " + img_path +  '/impro_out/ </br>'
	else:	
		result += "File: " + filter + "_processed_data </br>"
		result += "Path: " + img_path +  '/impro_out/ </br>'

	print result

        context = {'result': result}
	return StreamingHttpResponse(json.dumps(context), content_type='application/json')

#	context = transform(img_path,filter,canny_sigma)		
#	return HttpResponse(json.dumps(context), content_type='application/json')

    if option == '2' and os.path.isfile(dim_path) and dim_path != "":
	dim_red = request.GET.get("dim_red","pca").strip()
        dim_k = request.GET.get("dim_k",2)
	print "dim_path: " + dim_path + " dim_red: " + dim_red + " K: " + dim_k		

	r = threading.Thread(target=reduce, args=(dim_path,dim_red,dim_k), name='reduce')
	r.start()

	inputdir = os.path.dirname(dim_path)
	output_data = inputdir + "/" + dim_red + str(dim_k) + "_Data"
	print output_data
	dimfile = open(dim_path,'r')
	n_data = 0

	for file in dimfile:
		n_data += 1

	result = "Job Submitted! </br>"
	result += "Description: Processing " + dim_path + " with " + dim_red + " k = " + str(dim_k) + "</br>"
        result += "File: " + os.path.basename(output_data) + '</br>'
        result += "Path: " + os.path.dirname(output_data) +  '/' + dim_red + str(dim_k) + "_Features/" + '</br>'
        result += "Dimension: " + str(n_data) + " x " + str(dim_k) + "</br>"

	print result

        context = {'result': result}
	return StreamingHttpResponse(json.dumps(context), content_type='application/json')

#	context = reduce(dim_path,dim_red,dim_k)		
#	return HttpResponse(json.dumps(context), content_type='application/json')

    if option == '3' and os.path.isfile(clu_path) and clu_path != "":
	clu_alg = request.GET.get("clu_alg","kmeans").strip()
        clu_k = request.GET.get("clu_k",2)
	print "clu_path: " + clu_path + " clu_alg: " + clu_alg + " K: " + clu_k		

	c = threading.Thread(target=cluster, args=(clu_path,clu_alg,clu_k), name='cluster')
	c.start()

	inputdir = os.path.dirname(clu_path)
	output_data = inputdir + "/" + clu_alg + str(clu_k) + "_Data"
	print output_data
	clufile = open(clu_path,'r')
	n_data = 0

	for file in clufile:
		n_data += 1

	result = "Job Submitted! </br>"
	result += "Description: Processing " + clu_path + " with " + clu_alg + " k = " + str(clu_k) + "</br>"
        result += "File: " + os.path.basename(output_data) + '</br>'
        result += "Path: " + os.path.dirname(output_data) +  '/' + clu_alg + str(clu_k) + "_Features/" + '</br>'
        result += "Dimension: " + str(n_data) + " x " + str(clu_k) + "</br>"

	print result

        context = {'result': result}
	return StreamingHttpResponse(json.dumps(context), content_type='application/json')

#	context = reduce(dim_path,dim_red,dim_k)		
#	return HttpResponse(json.dumps(context), content_type='application/json')


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


#Clustering
    if  clu_path == "":
	clu_info = "please input a valid data file"
        data = {'clu_info': clu_info}
    
    elif not os.path.isfile(clu_path):
	clu_info = "invalid data file"
        data = {'clu_info': clu_info}
	return HttpResponse(json.dumps(data), content_type='application/json')
    
    else:
	n_data = 0
	n_feature = 0
	clu_flag = True
	clu_file = open(clu_path,'r')
        for line in clu_file:
		n_data += 1
		if n_data == 1:
			n_feature = len(line.split(" "))
        	else:
			if n_feature != len(line.split(" ")):
				clu_flag = False
				break
	clu_file.close()
	if clu_flag:		
		clu_info = "Data set " + str(n_data) + " x " + str(n_feature)
	else:
		clu_info = "invalid data file: unbalanced data dimension"
        data = {'clu_info': clu_info}
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

	n_data = 0
	n_features = 0
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
		
#		result = "processing . . . " + file
#               context = {'result': result}
#		yield context

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
#	yield context
#	return
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
	print "inputdir: " + inputdir + result
	inputfile = open(inputpath,'r')
	for line in inputfile:
                input_n = len(line.split(" "))
                n_data += 1
		#print "Selected data set has " + str(input_n) + " features"
                #break
        inputfile.close()

       # result = "File: " + os.path.basename(output_data) + '</br>'
       # result += "Path: " + os.path.dirname(output_data) +  '/' + alg + str(k) + "_Features/" + '</br>'
       # result += "Dimension: " + str(n_data) + " x " + str(n_features) + "</br>"
       # context = {'result': result}
       # yield context

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
                result += "Path: " + os.path.dirname(output_data) +  '/' + alg + str(k) + "_Features/" + '</br>'
                result += "Dimension: " + n_data + " x " + n_features + "</br>"
                result += "Size: " + file_size + ' bytes'
		print result
		sc.stop()		

        print "Dimension reduction finished!"

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
	
	if os.path.isdir(output_dir):
        	os.system("rm -r " + output_dir)
	
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

	return output_data


def cluster(inputpath,alg,k):
	from pyspark import SparkContext
        from pyspark.sql import SQLContext, Row
        from pyspark.mllib.linalg import Vectors
        from pyspark import SparkConf, SparkContext
	
	n_data = 0
	n_features = 0
	result = "successful!"
	inputdir = os.path.dirname(inputpath)
	print "inputdir: " + inputdir + result
	inputfile = open(inputpath,'r')
	for line in inputfile:
                input_n = len(line.split(" "))
                n_data += 1
		#print "Selected data set has " + str(input_n) + " features"
                #break
        inputfile.close()

       # result = "File: " + os.path.basename(output_data) + '</br>'
       # result += "Path: " + os.path.dirname(output_data) +  '/' + alg + str(k) + "_Features/" + '</br>'
       # result += "Dimension: " + str(n_data) + " x " + str(n_features) + "</br>"
       # context = {'result': result}
       # yield context

	if int(k) == 1:
                print "k should be greater than 1"
                result =  "k should be greater than 1"
	else:
		os.system("export _JAVA_OPTIONS='-Xms1g -Xmx40g'")
		conf = (SparkConf().set("spark.driver.maxResultSize", "5g"))
                sc = SparkContext(conf=conf)
                sqlContext = SQLContext(sc)
                lines = sc.textFile(inputpath).map(lambda x:x.split(" "))
                lines = lines.map(lambda x:(x[0],[float(y) for y in x[1:]]))
                df = lines.map(lambda x: Row(labels=x[0],features=Vectors.dense(x[1]))).toDF()
	
		if alg == "kmeans":
			output_data = kmeans(inputdir,df,alg,k)
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
                result += "Path: " + os.path.dirname(output_data) +  '/' + alg + str(k) + "_Features/" + '</br>'
                result += "Dimension: " + n_data + " x " + n_features + "</br>"
                result += "Size: " + file_size + ' bytes'
		print result
		sc.stop()		

        print "Clustering finished!"

        context = {'n_data': n_data, 'n_features': n_features, 'result': result}
	return context


def kmeans(inputdir,df,alg,k):
	from pyspark.ml.clustering import KMeans
        from numpy import array
        from math import sqrt	
	kmeans = KMeans(k=int(k), seed=1,initSteps=5, tol=1e-4, maxIter=20, initMode="k-means||", featuresCol="features")
        model = kmeans.fit(df)
        kmFeatures = model.transform(df).select("labels", "prediction")
	output_data = writeOutClu(inputdir,kmFeatures,alg,k)
	return output_data

def writeOutClu(inputdir,df,alg,k):
	output_dir = inputdir + "/" + alg + str(k) + "_Features"
	output_data = inputdir + "/" + alg + str(k) + "_Data"
	n_data = 0	
	n_features = 0
	
	if os.path.isdir(output_dir):
        	os.system("rm -r " + output_dir)
	
	df.rdd.repartition(1).saveAsTextFile(output_dir)
        outputfile = open(output_data, 'w')
        inputfile = open(output_dir + '/part-00000', 'r')
        for line in inputfile:
			n_data += 1
                        x = line.split("=")[2].split(")")[0]
                        y = line.split("'")[1]
                        outputfile.write(y + " " + x + '\n')
        inputfile.close()
        outputfile.close()

	return output_data


def test(a,b,c):
	print "HERE WE ARE!!!!"
