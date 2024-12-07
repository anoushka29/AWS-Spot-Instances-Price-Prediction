from pyspark.ml.feature import VectorAssembler
from pickle import TRUE
from flask import Flask, render_template, request
import numpy as np
from pyspark.ml.classification import DecisionTreeRegressionModel
from pyspark.sql.types import StructField, StructType, IntegerType
from pyspark import SparkContext
from collections.abc import MutableMapping
#from collections import MutableMapping

sc = SparkContext()


app = Flask(__name__)


@app.route('/')
def man():
    return render_template('home.html')


@app.route('/north')
def north():
    return render_template("north.html")


@app.route('/south')
def south():
    return render_template("south.html")


@app.route('/east')
def east():
    return render_template("east.html")


@app.route('/west')
def west():
    return render_template("west.html")


@app.route('/predict', methods=['POST'])
def home():
    da = request.form['a']
    data2 = int(request.form['Hour'])
    data3 = int(request.form['OS'])
    data1 = int(request.form['MONTH'])
    data4 = int(request.form['RegionIndex'])
    data5 = int(request.form['InstanceIndex'])
    acdata = sc.parallelize(
        [{'Month': data1, 'Hour': data2, 'OS': data3, 'Region': data4, 'Instance': data5}])

    schema = StructType([
        StructField('Month', IntegerType(), True),
        StructField('Hour', IntegerType(), True),
        StructField('OS', IntegerType(), True),
        StructField("Region", IntegerType(), True),
        StructField("Instance", IntegerType(), True)
    ])
    from pyspark import SQLContext
    sqlContext = SQLContext(sc)
    df = sqlContext.createDataFrame(acdata, schema)
    cols0 = ["Month", "Hour", "OS", "Region", "Instance"]
    vec_assm = VectorAssembler(inputCols=cols0, outputCol='features')
    test = vec_assm.transform(df)
    if da == "north":
        model_1 = DecisionTreeRegressionModel.load(
            "/home/masterhadoop/Desktop/AWS_SPOT_INSTANCE_APP/north_d")
        pred = model_1.transform(test)
        p = pred.select("prediction")
        c = p.toPandas()
        a = c.prediction.values[0]
        return render_template('after.html', msg=a)


if __name__ == "__main__":
    app.run(debug=TRUE)
