from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
from math import sqrt
from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName("K-Means in Spark")
sc= SparkContext(conf=conf)

data=sc.textFile("irisdata.txt")
parsedData=data.map(lambda line: array([(x) for x in line.split(',')]))

print(parsedData.take(10))

param=parsedData.map(lambda x: array([float(x[0]), float(x[1]), float(x[2]), float(x[3])]))

def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))
    WSSSE = param.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    print("Within Set Sum of Squared Error = " + str(WSSSE))
    
for i in range (0,1):
    clusters= KMeans.train(param, 1, maxIterations = 100, runs = 100, initializationMode="random")
    WSSSE = param.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    print("With " +str(1) + " clusters: Within Set Sum of Squared Error = " + str(WSSSE))
