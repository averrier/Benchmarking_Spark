#!/usr/bin/env python
# coding: utf-8

#TEST
#ATTENTION: pour nb workers variables (7, 5, 3, 1) ne faire que le test 0

print("START")

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
import time
import timeit 
from datetime import datetime
import sys

#GET NUMBER OF WORKERS
nb_workers = -1
if(len(sys.argv) == 2):
    nb_workers = int(sys.argv[1])    

#SPARK CONTEXT
sc = SparkContext.getOrCreate()
sql_sc = SQLContext(sc)

#LOAD DATASET
schema = StructType([StructField("age",LongType(),True),StructField("workclass",StringType(),True),StructField("fnlwgt",LongType(),True),StructField("education",StringType(),True),StructField("education_num",LongType(),True),StructField("marital_status",StringType(),True),StructField("occupation",StringType(),True),StructField("relationship",StringType(),True),StructField("race",StringType(),True),StructField("sex",StringType(),True),StructField("capital_gain",LongType(),True),StructField("capital_loss",LongType(),True),StructField("hours_per_week",LongType(),True),StructField("native_country",StringType(),True),StructField("income",StringType(),True)])


df = sql_sc.read.load("gs://example_adult/adult.csv", format="csv", sep=",", inferSchema="false", header="false", schema=schema, ignoreLeadingWhiteSpace="true")

stringCol = [f.name for f in df.schema.fields  if isinstance(f.dataType, StringType)]

expr = " and ".join("%s != '?'" % col for col in stringCol)

df = df.filter(expr)
cols = df.columns
                     
#LIST IMPLEMENTED TESTS
#-1 : NO ID
#0 : NUMBER OF ROWS = 100%
#1 : NUMBER OF ROWS = 75%
#2 : NUMBER OF ROWS = 50%
#3 : NUMBER OF ROWS = 25%
#4 : FEATURES = {age, workclass, fnlwgt, education, marital_status, occupation, sex, hours_per_week, native_country, income}
#5 : FEATURES = {age, workclass, fnlwgt, occupation, hours_per_week, native_country, income}
#6 : FEATURES = {workclass, hours_per_week, native_country, income}
#implemented_tests = [0]
implemented_tests = [0, 1, 2, 3, 4, 5, 6]
testID = -1
metrics = []
models_settings = [[LogisticRegression(featuresCol = 'features', labelCol = 'label'), "income", BinaryClassificationEvaluator()],
                    [RandomForestClassifier(labelCol="label", featuresCol="features"), "native_country", MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")],
                    [LinearRegression(labelCol="label", featuresCol="features", regParam=0.3, elasticNetParam=0.8, maxIter=100), "hours_per_week", RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")]
                    #,[RandomForestRegressor(labelCol="label", featuresCol="features"), "hours_per_week", RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")]
                    ]

for model, targetCol, evaluator in models_settings:
    for testID in implemented_tests:
        #SET THE TEST SETTING
        if(testID == -1):
            print("please provide the test ID")
        elif(testID == 0):
            print("#0 : NUMBER OF ROWS = 100%, MODEL =", model)
            df_test = df

        elif(testID == 1):
            print("#1 : NUMBER OF ROWS = 75%, MODEL =", model)
            df_test = df.sample(False, 0.75)

        elif(testID == 2):
            print("#2 : NUMBER OF ROWS = 50%, MODEL =", model)
            df_test = df.sample(False, 0.5)

        elif(testID == 3):
            print("#3 : NUMBER OF ROWS = 25%, MODEL =", model)
            df_test = df.sample(False, 0.25)

        elif(testID == 4):
            print("#4 : FEATURES = 10/15, MODEL =", model)
            df_test = df.drop("education_num").drop("relationship").drop("race").drop("capital_gain").drop("capital_loss")

        elif(testID == 5):
            print("#5 : FEATURES = 7/15, MODEL =", model)
            df_test = df.drop("education_num").drop("relationship").drop("race").drop("capital_gain").drop("capital_loss").drop("marital_status").drop("education").drop("sex")

        elif(testID == 6):
            print("#6 : FEATURES = 4/15, MODEL =", model)
            df_test = df.drop("education_num").drop("relationship").drop("race").drop("capital_gain").drop("capital_loss").drop("marital_status").drop("education").drop("sex").drop("age").drop("fnlwgt").drop("occupation")


        cols = df_test.columns
        stringCol = [f.name for f in df_test.schema.fields  if isinstance(f.dataType, StringType)]
        categoricalColumns = stringCol
        numericCols = [col for col in cols if col not in stringCol]

        features_indexers = [
            StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c), handleInvalid="keep")
            for c in categoricalColumns if c != targetCol
        ]    

        features_encoder = OneHotEncoderEstimator(
            inputCols=[indexer.getOutputCol() for indexer in features_indexers],
            outputCols=[
                "{0}_encoded".format(indexer.getOutputCol()) for indexer in features_indexers],
            handleInvalid="keep"
        )   

        features_assembler = VectorAssembler(
            inputCols=features_encoder.getOutputCols()+[col for col in numericCols if col != targetCol],#[indexer.getOutputCol() for indexer in indexers]+numericCols,
            outputCol="features"
        )

        stages = features_indexers + [features_encoder, features_assembler]

        if(targetCol in categoricalColumns):
            label_indexer = StringIndexer(inputCol = targetCol, outputCol = 'label', handleInvalid="keep")
            stages += [label_indexer]
        else:
            df_test = df_test.withColumn("label", df_test[targetCol])

        stages += [model]

        train, test = df_test.randomSplit([0.7, 0.3], seed = 2020)

        pipeline = Pipeline(stages = stages)

        #train_startTime = time.clock()
        train_startTime = datetime.now()
        model_fit = pipeline.fit(train)
        train_endTime = datetime.now()
        #train_endTime = time.clock()

        #prediction_startTime = time.clock()
        prediction_startTime = datetime.now()
        predictions = model_fit.transform(test)
        prediction_endTime = datetime.now()
        #prediction_endTime = time.clock()
        
        train_time = train_endTime-train_startTime
        train_time = train_time.total_seconds()
        prediction_time = prediction_endTime-prediction_startTime
        prediction_time = prediction_time.total_seconds()

        nb_rows = df_test.count()
        nb_columns = len(df_test.columns)
        #print("Number of rows: ", nb_rows)
        #print("Time: ", ex_time)
        #df_test.printSchema()
        #predToShow = predictions.select("prediction", "label", targetCol).collect()
        #print(predToShow[100])
        #print(predToShow[1000])
        #print(predToShow[5000])

        accuracy = evaluator.evaluate(predictions)

        metrics.append((str(nb_workers), str(testID), str(nb_rows), str(nb_columns), str(train_time), str(prediction_time), str(accuracy)))
print(metrics)
#df_results = sql_sc.read.load("gs://example_adult/results.csv", format="csv", sep=",", inferSchema="true", header="true", ignoreLeadingWhiteSpace="true")
df_metrics = sql_sc.createDataFrame(metrics)
#df_results = df_results.union(newRow)
df_metrics.coalesce(1).write.mode("overwrite").csv("gs://example_adult/results_"+str(nb_workers)+"workers.csv")