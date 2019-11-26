"""

Run with:
  bin/spark-submit <path>

This program requires NumPy (http://www.numpy.org/).
"""

from __future__ import print_function

import sys
import logging
from pyspark.sql import SparkSession
import numpy as np
from numpy import array
from math import sqrt
from pyspark import SparkContext
from datetime import datetime
import time
#from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.sql.types import DoubleType, FloatType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans, KMeansModel
import spark
from utils import innolog
import os
import codecs
import argparse


#PATH = os.path.abspath(os.path.join(os.getcwd(), "."))
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
print(PATH)

logger = logging.getLogger('bikes_clustering')
logger.addHandler(innolog.get_handler(os.path.join(PATH,'model','bikes_clustering.log')))
logger.setLevel(logging.INFO)

def parseVector(line):
    return np.array([float(x) for x in line.split(' ')])

def json_dataset(spark,path):
    sc = spark.sparkContext
    # A JSON dataset is pointed to by path.
   
    #peopleDF = spark.read\
    #                .option("multiLine", true)\
    #                .option("mode", "PERMISSIVE")\
    #                .json(path)

    dataRDD = spark.read\
                    .json(sc.wholeTextFiles(path)\
                    .values())
    
    print("RDD")
    dataRDD.show(10)

    # The inferred schema can be visualized using the printSchema() method
    dataRDD.printSchema()
    
    # Creates a temporary view using the DataFrame
    dataRDD.createOrReplaceTempView("table_data")

    # SQL statements can be run by using the sql methods provided by spark
    dataDF = spark.sql("SELECT * FROM table_data \
            where longitude not like '%not%relevant%' and latitude not like '%not%relevant%'")
    print('DF')
    #dataDF.show(10)
    return dataRDD, dataDF

def cast_to_float(column):
    return column.cast('float')

def cast_df_tofloat(df,column1,column2):
    return df.select(df.column1.cast('float'),df.column2.cast('float'))

def cast_DFv2_tofloat(df,column1):
    return df.withColumn(column1, df[column1].cast(FloatType()))

def cleaning_RDD_pandas (df_):
    df = df_.toPandas()
    print(df.iloc[:10,:])
    print(df.columns)
    # cleaning steps #
    # keeping only rows with coordinates / latitude or longitude not empty value
    df_1 = df[df['coordinates'].notnull() | (df['longitude'].notnull() & df['latitude'].notnull())]

    # retrieving only Rows with valid float type in fields longitude and latitude
    # after analysing the data content, we notice the noise value 'not relevant' in longitude and latitudes ==> 2 rows
    df_1 = df_1[df_1['longitude'].apply(lambda x: type(x) in [int, np.int64, float, np.float64])]
    df_1 = df_1[df_1['latitude'].apply(lambda x: type(x) in [int, np.int64, float, np.float64])]

    # Filling the longitude / latitude value with the value in 'coordinates' when exists and are valid
    ### # checking the Nan rows to replace in longitude and latitude when coordinates exists ==> creating first temporary filtrering dataframe
    tmp_df = df_1[df_1['coordinates'].notnull() & (df_1['longitude'].isnull() | df_1['latitude'].isnull())]

    df_1['latitude'].fillna(tmp_df['coordinates'].apply(lambda x : x['latitude']),inplace=True)
    df_1['longitude'].fillna(tmp_df['coordinates'].apply(lambda x : x['longitude']),inplace=True)

    ## Filling Nan Value with valid value in "Coordinates" from 'longitude' and "latitude" fields
    df_1['coordinates'].fillna(df_1.apply(lambda x : {'latitude':x['latitude'],'longitude':x['longitude']},axis=1),inplace=True)
    
    ## Keeping only Rows with valid float type in Coordinates ==> May be escape if coordinates is not needed for our training steps
    ## hence the df_1 will be sufficient.
    #df_2 = df_1[df_1.coordinates.apply(lambda x: type(x['latitude']) in [int, np.int64, float, np.float64])]
    #df_2 = df_1[df_1.coordinates.apply(lambda x: type(x['longitude']) in [int, np.int64, float, np.float64])]

    return df_1



if __name__ == "__main__":

    #spark = SparkSession \
    #    .builder \
    #    .appName("Python Spark SQL data source example") \
    #    .getOrCreate()
    spark = spark.get_spark()

    if len(sys.argv) <4 or len(sys.argv) >5:
        print("Usage: kmeans <input file> <mode(trainning or predicting)> <k> <output file> OR kmeans <input file> <mode(trainning or predicting)> <k> ", file=sys.stderr)
        sys.exit(-1)

    mode = sys.argv[2] # mode is 'trainning' or 'predicting'
    k = int(sys.argv[3]) # Number of cluster needed for clustering
    currentdate = datetime.now().strftime("%Y-%m-%d")

    ###### LOADING DATA #####
    # INPUT
    input_data_path = os.path.join(PATH, "datasets","input", sys.argv[1])
    output_data_path = os.path.join(PATH, "datasets","output")
    datapath_input = input_data_path
    datapath_output = output_data_path
    modelpath = os.path.join(PATH ,'model','KMeansModel')
    modelpath_archives = os.path.join(PATH ,'model','archives','KMeansModel-'+currentdate)

     # OUTPUT
    if len(sys.argv) ==5:
        datapath_output_file1 = os.path.join(datapath_output, sys.argv[3])
        datapath_output_file2 = os.path.join(datapath_output,'clusters_groups_count-') 
    else:
        if len(sys.argv) ==4:
            datapath_output_file1 = os.path.join(datapath_output ,'clustering_assignment_by_location-')
            datapath_output_file2 = os.path.join(datapath_output ,'clusters_groups_count-')
    #print(datapath_input, datapath_output)

    dataRDD, dataDF = json_dataset(spark,datapath_input)
  
    #### 1st methods of extracting features suitable for MLLIB library
    #features = dataDF['latitude','longitude']
    #features = cast_DFv2_tofloat(features,'latitude')
    #features = cast_DFv2_tofloat(features,'longitude')
    #features.printSchema()
    #features = features.rdd.map(list)
    #print('FEATURE',features.take(2)) # print RDD content limit to 2 rows
 
    #### 2nd methods of extracting features for ML library
    dataDF = cast_DFv2_tofloat(dataDF,'latitude')
    dataDF = cast_DFv2_tofloat(dataDF,'longitude')

    # Adding Features columns in datasets that will be used by the model for training 
    assembler = VectorAssembler(inputCols=["latitude", "longitude"],outputCol="features")
    dataDF=assembler.transform(dataDF)
    #dataDF.show()

    if mode == "training":
       
        # splitting datasets for training and testing
        (training, testdata) = dataDF.randomSplit([0.7, 0.3], seed = 5043)
        kmeans = KMeans().setK(k)
        model = kmeans.fit(training)

        # Predicting the cluster that each id will belong 
        transformed=model.transform(testdata).withColumnRenamed("prediction","cluster_id")

        #archives old model
        model_old = KMeansModel.load(modelpath)
        model_old.write().overwrite().save(modelpath_archives)
        logger.info('Old Daily Clustering Bikes by location Model has been archived on the {}'.format(datetime.now()))
        ##### Save model
        model.write().overwrite().save(modelpath)
        logger.info('New Daily Clustering Bikes by location Model has been trained on the {}'.format(datetime.now()))
    
    if mode == "predicting":
        model = KMeansModel.load(modelpath)
        logger.info('Daily Clustering Bikes by location Started on the {} '.format(datetime.now()))

        transformed_tot=model.transform(dataDF).withColumnRenamed("prediction","cluster_id")

        #creating output dataset
        transformed_tot.createOrReplaceTempView("data_table")
        data_table = spark.sql("SELECT data_table.id, data_table.name, data_table.address, data_table.latitude, data_table.longitude, data_table.position, data_table.coordinates, data_table.cluster_id FROM data_table")
        data_table.coalesce(1).write.format('json').save(datapath_output_file1 + currentdate + '.json')
        logger.info('Daily Output file {} generated on {}'.format(datapath_output_file1 + currentdate +'.json', datetime.now()))
        transformed_tot.cache()

        # creating the centroides grouping and output file for this gouping
        centerList=list()
        centers1 = model.clusterCenters()
        count=int()
        for center in centers1:
            temp_list=center.tolist()
            temp_list.append(count)
            centerList.append(temp_list)
            count=count+1       
        centers=spark.createDataFrame(centerList)
        centers.createOrReplaceTempView("centers")
        resultsDFF = spark.sql("SELECT centers._1 as latitude, centers._2 as longitude FROM data_table, centers WHERE data_table.cluster_id=centers._3")
        #centers.show()
        data=resultsDFF.groupBy("latitude", "longitude").count()
        data.coalesce(1).write.format('json').save(datapath_output_file2 + currentdate +'.json')
        logger.info('Daily Output file {} generated on {} '.format(datapath_output_file2 + currentdate +'.json', datetime.now()))

    else:
        print("Wrong Mode Entry !!!!! , please enter mode value 'training' or 'predicting' !!!! ", file=sys.stderr)
        sys.exit(-1)
    logger.info('Daily Clustering Bikes by location Terminated on the {} '.format(datetime.now()))
    spark.stop()
    
