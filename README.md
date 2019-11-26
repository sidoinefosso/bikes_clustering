# bikes_clustering
This project is a clustering project of the geographical location of bikes in brisbane cities.

the program has been developed in python3 and the clustering algorith is based on pyspark ML library.
the program will run in a distributed spark cluster, for my case, i've used a standalone spark instance.

there are couple of folders that need to be described:
1 - model : contains the executables python file (.py), the saved Models, the archives's folder where the model will be archives in case of new training, the log file where the information about daily job are saved, and an utils folder which contain a logging module
2 - datasets : contains 2 folders:
              - input : input data (the daily data should be copy or landed every day to this folder).
              - output: the output file containing the clustering results in 'json' and another file which is a simple count of               cluster group bikes contains.
3 - test (toDO) : the test folder should contains the unitary test.py that will test the code.

# HOW TO LAUNCH THE PROGRAM

the program can be launch manualy or scheduled in a plateform as a daily Job.
there exists many solution of scheduling and monitoring these daily script job, either localy in the server where the script will be launched (cron job, Apache Nifi, Oozie etc...) or on the cloud depending on cloud provider (Azure Data factory, AWS GLUE, etc...).

for our case, we will chose to schedul the job locally in a linux server in a user "crontab table".

# input files: Brisbane_CityBike.json

# output files : clustering_assignment_by_location-2019-11-26.json / clusters_groups_count-2019-11-26.json

# Launching Command and Arguments.

<SPARK_HOME>/bin/spark-submit --master <spark-server-url:local[4]> <project path>/bikes_clustering/model/cluster_model.py <input file data: Brisbane_CityBike.json> <mode:training or predicting> <k:clustering size> <optional:output filename> 

there is at list 4 and maximum of 5 (optional) arguments that can be passed to the launching command:
  # 1- <SPARK_HOME>/bin/spark-submit --master <spark-server-url:local[4]> 
  this the spark launching, we can pass more arguments with --conf (see spark documentation) as per our requirements in order     to size the driver, the executors, the CPU, the JVM. here, i ran the script locally with 4 Cores.
  # 2- the first argument is the input data file in 'json' format.
  # 3- the second arguments is the main program file entire path ./../cluster_model.py
  # 4- the third argument is the launching mode : training or predicting (for daily run, we will use predicting mode, training mode can be used if the model need to be trained again)
  # 5- the fourth argument is the cluster size needed k.
  # 6- the fifht (optional) argument is the outputfile name.
  
  # scheduling for daily ingestion every day at 18.00
  00 18 * * * <SPARK_HOME>/bin/spark-submit --master <spark-server-url:local[4]> <project path>/bikes_clustering/model/cluster_model.py <input file data: Brisbane_CityBike.json> <mode:training or predicting> <k:clustering size> <optional:output filename>
  
  # ToDO
  
  Remaining task:
    1- more cleaning of dataset and input dataset enrichment
    2- normalizing the log for production usage
    3- setting up a framework like Apache nifi for Job scheduling and spark job submitting
    4 - tuning the clustering model
    5 - deploying the tests case
    6 - packing the python3 environement and dependancies  with docker so that it's could be launched on every server.
    
  
  
