import os
import sys
import re
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.mllib.linalg import Vectors
from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import PCA

inputpath = sys.argv[1]
k = sys.argv[2]

inputdir = os.path.dirname(inputpath)
print "inputdir: " + inputdir


os.system("export _JAVA_OPTIONS='-Xms1g -Xmx40g'")
conf = (SparkConf().set("spark.driver.maxResultSize", "5g"))
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
lines = sc.textFile(inputpath).map(lambda x:x.split(" "))
lines = lines.map(lambda x:(x[0],[float(y) for y in x[1:]]))
df = lines.map(lambda x: Row(labels=x[0],features=Vectors.dense(x[1]))).toDF()






####Run####
pca = PCA(k=int(k),inputCol="features", outputCol="pca_features")
model = pca.fit(df)
outData = model.transform(df)
pcaFeatures = outData.select("labels","pca_features")

####Write Out####
output_dir = inputdir + "/pca" + str(k) + "_Features"
output_data = inputdir + "/pca" + str(k) + "_Data"
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

print "Dimension reduction finished!"
