package org.apache.spark.ml.knn

import org.apache.log4j._
import org.apache.spark.ml.classification.{OneVsRest, RandomForestClassifier}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{IntegerType, _}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.collection.mutable.ArrayBuffer
import scala.collection.parallel.immutable.ParSeq
import scala.util.Random

object KDClustering {

  def convertFeaturesToVector(df: DataFrame): DataFrame = {
    val spark = df.sparkSession
    import spark.implicits._
    val convertToVector = udf((array: Seq[Double]) => {
      Vectors.dense(array.map(_.toDouble).toArray)
    })

    df.withColumn("features", convertToVector($"features"))
  }


  def getPredictedLabel(labels: Array[Int]): Int ={
    val mp: Map[Int, Int] = labels.groupBy(identity).mapValues(_.size)
    if(mp.isEmpty) {
      -1
    }
    else {
      val mapResult = mp.maxBy { case (key, value) => value }
      mapResult._1
    }
  }

  def isCorrectPrediction(result: (Int, Int)): Int = {
    if (result._1 == result._2) {
      1
    }
    else {
      0
    }
  }

  def formatResults(x: String): Int ={
    val index = x.toString.indexOf(",")
    val label = x.toString.substring(1, index).toInt // get the label from the [label, [features]] element
    label
  }

  def maxValue(a: Double, b:Double): Double ={
    if(a >= b) { a }
    else { b }
  }

  def calculateClassifierResults(distinctClasses: DataFrame, confusionMatrix: DataFrame): Array[String]={//String ={
    import distinctClasses.sparkSession.implicits._
    //FIXME - don't calculate twice

    val classLabels = distinctClasses.collect().map(x => x.toSeq.last.toString.toDouble.toInt)

    val maxLabel: Int = classLabels.max
    val minLabel: Int = classLabels.min
    val numberOfClasses = classLabels.length
    val classCount = confusionMatrix.columns.length - 1
    val testLabels = distinctClasses.map(_.getAs[Int]("label")).map(x => x.toInt).collect().sorted

    val rows = confusionMatrix.collect.map(_.toSeq.map(_.toString))
    val totalCount = rows.map(x => x.tail.map(y => y.toDouble.toInt).sum).sum
    val classMaps = testLabels.zipWithIndex.map(x => (x._2, x._1))

    var AvAvg = 0.0
    var MAvG = 1.0
    var RecM = 0.0
    var PrecM = 0.0
    var Precu = 0.0
    var Recu = 0.0
    var FbM = 0.0
    var Fbu = 0.0
    var AvFb = 0.0
    var CBA = 0.0

    var tSum = 0.0
    var pSum = 0.0
    var tpSum = 0.0
    val beta = 0.5 // User specified

    for (clsIndex <- minLabel to maxLabel - minLabel) {
      val colSum = rows.map(x => x(clsIndex + 1).toInt).sum
      val rowValueSum = if (classMaps.map(x => x._2).contains(clsIndex)) rows.filter(x => x.head.toDouble.toInt == clsIndex)(0).tail.map(x => x.toDouble.toInt).sum else 0

      val tp: Double = if (classMaps.map(x => x._2).contains(clsIndex)) rows.filter(x => x.head.toDouble.toInt == clsIndex)(0).tail(clsIndex).toDouble.toInt else 0
      val fn: Double = colSum - tp
      val fp: Double = rowValueSum - tp
      val tn: Double = totalCount - tp - fp - fn

      val recall = tp / (tp + fn)
      val precision = tp / (tp + fp)

      AvAvg += ((tp + tn) / (tp + tn + fp + fn))
      MAvG *= recall
      RecM += { if(recall.isNaN) 0.0 else recall }
      PrecM += precision

      val getAvFb: Double= {
        val result = ((1 + Math.pow(beta, 2.0)) * precision * recall) / (Math.pow(beta, 2.0) * precision + recall)
        if(result.isNaN) {
          0.0
        }
        else result
      }
      AvFb += getAvFb

      val rowColMaxValue = maxValue(colSum, rowValueSum)
      if(rowColMaxValue > 0) {
        CBA += tp / rowColMaxValue
      }
      else {
      }

      // for Recu and Precu
      tpSum += tp
      tSum += (tp + fn)
      pSum += (tp + fp)
    }

    AvAvg /= classCount
    MAvG = {  val result = Math.pow(MAvG, 1/numberOfClasses.toDouble); if(result.isNaN) 0.0 else result }
    RecM /= classCount
    PrecM /= classCount
    Recu = tpSum / tSum
    Precu = tpSum / pSum
    FbM = { val result = ((1 + Math.pow(beta, 2.0)) * PrecM * RecM) / (Math.pow(beta, 2.0) * PrecM + RecM); if(result.isNaN) 0.0 else result }
    Fbu = { val result = ((1 + Math.pow(beta, 2.0)) * Precu * Recu) / (Math.pow(beta, 2.0) * Precu + Recu); if(result.isNaN) 0.0 else result }
    AvFb /= classCount
    CBA /= classCount

    Array(AvAvg.toString, MAvG.toString, RecM.toString, PrecM.toString, Recu.toString, Precu.toString, FbM.toString, Fbu.toString, AvFb.toString, CBA.toString)
  }

  def smoteSample(randomInts: Random, currentClassZipped: Array[Row], cls: Int): Row = {
    def r = randomInts.nextInt(currentClassZipped.length)

    val sampled = Array(currentClassZipped(r), currentClassZipped(r), currentClassZipped(r), currentClassZipped(r), currentClassZipped(r))
    val values: Array[Array[Double]] = sampled.map(x=>x(2).asInstanceOf[DenseVector].toArray)
    val tp: Array[Double] = values.transpose.map(_.sum /values.length)
    Row(0L, cls,  Vectors.dense(tp.map(_.toDouble)))
  }

  def smote(spark: SparkSession, df: DataFrame, presentClass: Int, numSamples: Int): DataFrame = {
    df.show()
    println("Previous DF Schema")
    val randomInts: Random = new scala.util.Random()
    val currentCount = df.count()
    println("CLS: " + presentClass)

    println("numSamples: " + numSamples)
    val finaDF = if (currentCount < numSamples) {
      val samplesToAdd = numSamples - currentCount
      println("Samples to add: " + samplesToAdd)
      val collected: Array[Row] = df.collect()
      val results = (1 to samplesToAdd.toInt).map(x=>smoteSample(randomInts, collected, presentClass))

      val mappedResults: RDD[Row] = spark.sparkContext.parallelize(results)
      val mappedDF = spark.sqlContext.createDataFrame(mappedResults, df.schema)

      val joinedDF = df.union(mappedDF)
      println("   ~~~ NEW SCHEMA ~~~")
      joinedDF.printSchema()
      joinedDF
    }
    else {
      df
    }
    finaDF
  }

  def overSample(spark: SparkSession, df: DataFrame, numSamples: Int): DataFrame = {
    var samples = Array[Row]() //FIXME - make this more parallel
    //FIXME - some could be zero if split is too small
    val currentCount = df.count()
    if (0 < currentCount && currentCount < numSamples) {
      val currentSamples = df.sample(true, (numSamples - currentCount) / currentCount.toDouble).collect()
      samples = samples ++ currentSamples
    }

    val foo = spark.sparkContext.parallelize(samples)
    val x = spark.sqlContext.createDataFrame(foo, df.schema)
    df.union(x).toDF()
  }

  def getCountsByClass(spark: SparkSession, label: String, df: DataFrame): DataFrame = {
    val numberOfClasses = df.select("label").distinct().count()
    val aggregatedCounts = df.groupBy(label).agg(count(label)).take(numberOfClasses.toInt) //FIXME

    val sc = spark.sparkContext
    val countSeq = aggregatedCounts.map(x => (x(0).toString, x(1).toString.toInt)).toSeq
    val rdd = sc.parallelize(countSeq)

    spark.createDataFrame(rdd)
  }

  def sampleDataParallel(spark: SparkSession, df: DataFrame, presentClass: Int, samplingMethod: String, underSampleCount: Int, overSampleCount: Int, smoteSampleCount: Int): DataFrame = {
    println("*********")
    df.show()
    println("*********")
    val filteredDF2 = samplingMethod match {
      //case "undersample" => underSample(spark, df, underSampleCount)
      case "oversample" => overSample(spark, df, overSampleCount)
      case "smote" => smote(spark, df, presentClass, smoteSampleCount)
      case _ => df
    }
    filteredDF2
  }

  def sampleData(spark: SparkSession, df: DataFrame, samplingMethod: String): DataFrame = {
    val d = df.select("label").distinct()
    val presentClasses: Array[Int] = d.select("label").rdd.map(r => r(0)).collect().map(x=>x.toString.toInt)

    val counts = getCountsByClass(spark, "label", df)
    counts.show()
    val maxClassCount = counts.select("_2").agg(max("_2")).take(1)(0)(0).toString.toInt
    val minClassCount = counts.select("_2").agg(min("_2")).take(1)(0)(0).toString.toInt

    val overSampleCount = maxClassCount
    val underSampleCount = minClassCount
    val smoteSampleCount = maxClassCount
    println("Smote max count: " + maxClassCount)

    val myDFs: Array[(Int, DataFrame)] = presentClasses.map(x=>(x, df.filter(df("label") === x).toDF()))
    val dfs = presentClasses.map(x => sampleDataParallel(spark, myDFs.filter(y=>y._1 == x)(0)._2, x, samplingMethod, underSampleCount, overSampleCount, smoteSampleCount))

    println("Final count ")

    val all = dfs.reduce(_ union  _)
    println("^^^^^^^^^^^^")

    all
  }

  def smoteSample(cls: Int, currentCount: Long, clusterIds: Seq[Int], clusteredData: Map[Int, Seq[(Int, Int, Int, DenseVector)]], clusterCount: Int, randomInts: Random): Row = {
    val clusterIndex = clusterIds(randomInts.nextInt(clusterIds.length))
    val sampled: Array[(Int, Int, Int, DenseVector)] = Array.fill(5)(clusteredData(clusterIndex)(randomInts.nextInt(clusteredData(clusterIndex).length)))

    val values: Array[Array[Double]] = sampled.map(x=>x._4.asInstanceOf[DenseVector].toArray)

    val transposed: Array[Double] = values.transpose.map(_.sum /values.length)
    val r2 = Row(0, cls,  Vectors.dense(transposed.map(_.toDouble)))
    r2
  }

  def sampleDataSmotePlus(spark: SparkSession, df: DataFrame, samplingMethod: String, clusterKValue: Int, clusterResults: Map[Int,DataFrame]): DataFrame = {
    val d = df.select("label").distinct()
    println("^^^^^^^ distinct classes ^^^^^^^^^")
    d.show()
    println("^^^^^^^ distinct classes ^^^^^^^^^")
    val presentClasses: Array[Int] = d.select("label").rdd.map(r => r(0)).collect().map(x=>x.toString.toInt)

    val counts = getCountsByClass(spark, "label", df)
    println("***^ COUNTS OUTSIDE")
    counts.show()
    val maxClassCount = counts.select("_2").agg(max("_2")).take(1)(0)(0).toString.toInt
    val smoteSampleCount = maxClassCount
    var dfs: Array[DataFrame] = Array[DataFrame]()

    for (l <- presentClasses) {
      val currentCase = df.filter(df("label") === l).toDF()
      val filteredDF2 = samplingMethod match {
        case "smotePlus" => smotePlus(spark, currentCase, smoteSampleCount, clusterResults(l), l ,clusterKValue)
        case _ => currentCase
      }
      dfs = dfs :+ filteredDF2
    }

    val all = dfs.reduce(_ union  _)
    all
  }

  def smotePlus(spark: SparkSession, df: DataFrame, numSamples: Int, predictions: DataFrame, l: Int, clusterCount: Int): DataFrame = {
    val randomInts: Random = new scala.util.Random(42L)

    var samples = ArrayBuffer[Row]() //FIXME - make this more parallel
    val currentCount: Long = df.count()
    println("***^ INSIDE current count: "  + l + " " + currentCount)
    if (currentCount < numSamples) {
      val samplesToAdd = numSamples - currentCount

      println("At spark Means " + samplesToAdd)

      val predictionsCollected = predictions.collect().map(x=>(x(0).toString.toInt, x(1).toString.toInt, x(2).toString.toInt, x(3).asInstanceOf[DenseVector])).toSeq
      val clusteredData: Map[Int, Seq[(Int, Int, Int, DenseVector)]] = predictionsCollected.groupBy {_._1}
      val clusterIds = clusteredData.keySet.toSeq  //.map(x=>x._1).toSeq

      println("samplesToAdd: "  + samplesToAdd)

      val t01 = System.nanoTime()

      val addedSamples: ParSeq[Row] = (1 to samplesToAdd.toInt).par.map { _ => smoteSample(l: Int, currentCount: Long, clusterIds: Seq[Int], clusteredData: Map[Int, Seq[(Int, Int, Int, DenseVector)]], clusterCount: Int, randomInts: Random) }
      samples = samples ++ addedSamples
      val t11 = System.nanoTime()
      println("--------- LOOP TIME: " + (t11 - t01) / 1e9 + "s")
    }
    else {
      // we already have enough samples, skip
    }

    df.show()
    val tX = System.nanoTime()
    val currentArray = df.rdd.map(x=>Row(x(0), x(1), x(2).asInstanceOf[DenseVector])).collect()

    samples = samples ++ currentArray

    val samplesRestructured = samples.map(x=>x.toSeq).map(x=>(x(0).toString.toInt, x(1).toString.toInt, x(2).asInstanceOf[DenseVector]))
    val samplesParallelized = spark.createDataFrame(spark.sparkContext.parallelize(samplesRestructured))
    val result = samplesParallelized.withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "features")

    val tY = System.nanoTime()
    println("--------- Combine Time: " + (tY - tX) / 1e9 + "s")
    result
  }

  /******************************************/
  def getClassClusters(spark: SparkSession, l: Int, df: DataFrame, clusterKValue: Int): (Int, DataFrame) = {
    val result = if(clusterKValue < 2) {
      val currentCase = df.filter(df("label") === l).toDF()
      (l, currentCase)
    }
    else {
      val currentCase = df.filter(df("label") === l).toDF()
      val kmeans = new KMeans().setK(clusterKValue).setSeed(1L)
      val convertedDF = currentCase
      val model2 = kmeans.fit(convertedDF)
      // Make predictions
      println("^^^^^^ cluster count: " + model2.clusterCenters.length)
      val predictions = model2.transform(convertedDF).select("prediction", "index", "label", "features").persist()
      (l, predictions)
    }
    result
  }

  /******************************************/

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val input_directory = args(0)
    val output_directory = args(1)

    val datasetIn = args(2)
    val dataSize = args(3)
    val algorithmsTypes = args(4)

    val numFoldsParam = args(5)
    val treeSplits = args(6)
    val treeSplitMethod = args(7)
    val partitions = args(8)
    val numRFs = args(9)

    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._

    var resultArray: Array[Array[String]] = Array()
    val dataset = datasetIn

    val df = spark.read.
          option("inferSchema", true).
          option("header", true).
          csv(input_directory + "/" + dataset + "/" + dataset + dataSize + ".csv")

    val dfCount = df.count()

    val clusterCount = 5

      /// FIXME - doesnt seem to work if array size is > 1
      val leafSize =10///(dfCount / clusterCount).toInt //s = Array(100)//,25,100,250,1000)//, 5)
      //,10,25,50,100,500,1000,2500,5000,10000)
      val kValues = Array(5)//, 5) //,5,10,25,50,100) //and sqrt(n/2)
      val algorithms = if(algorithmsTypes=="all") {
        Array("none", "oversample", "smote", "smotePlus")
      }
      else if(algorithmsTypes=="smotes"){
        Array("smote", "smotePlus")
      }
      else if(algorithmsTypes=="smotePlus") {
        Array("smotePlus")
      }
      else {
        Array()
      }
      val isAllNN = false
      println("df count: " + dfCount)

      val train_index = df.rdd.zipWithIndex().map({ case (x, y) => (y, x) }) //.cache()

      val data = train_index.map({ r =>
        val array = r._2.toSeq.toArray.reverse
        val cls = array.head.toString.toDouble.toInt
        val rowMapped: Array[Double] = array.tail.map(_.toString.toDouble)
        //NOTE - This needs to be back in the original order to train/test works correctly
        (r._1, cls, rowMapped.reverse)
      })

      val results: DataFrame = data.toDF().withColumnRenamed("_1", "index")
        .withColumnRenamed("_2", "label")
        .withColumnRenamed("_3", "features")

      val converted: DataFrame = convertFeaturesToVector(results)

      val enableDataScaling = true
      val scaledDataInitial: DataFrame = if (enableDataScaling) {
        val scaler = new MinMaxScaler()
          .setInputCol("features")
          .setOutputCol("scaledFeatures")
        val scalerModel = scaler.fit(converted)
        val scaledData: DataFrame = scalerModel.transform(converted)
        scaledData.drop("features").withColumnRenamed("scaledFeatures", "features")
      } else {
        converted
      }.cache()

    val scaledDataX = scaledDataInitial.orderBy(rand())
    scaledDataX.printSchema()

    val schema = new StructType()
      .add(StructField("id", LongType, true))
      .add(StructField("index", LongType, true))
      .add(StructField("label", IntegerType, true))
      .add(StructField("features", VectorType, true))

    val scaledData = spark.createDataFrame(scaledDataX.rdd.zipWithIndex().map(x=>Row(x._2, x._1(0), x._1(1), x._1(2))), schema).drop("index").withColumnRenamed("id", "index")
    scaledData.show()

    val distinctClasses = scaledData.select("label").distinct()
    val maxLabel: Int = distinctClasses.collect().map(x => x.toSeq.last.toString.toDouble.toInt).max
    val minLabel: Int = distinctClasses.collect().map(x => x.toSeq.last.toString.toDouble.toInt).min

    val numFolds = numFoldsParam.toInt
      val counts = scaledData.count()
      var cuts = Array[Int]()

      cuts :+= 0
      if (numFolds < 2) {
        cuts :+= (counts * 0.2).toInt
      }
      else {
        for (i <- 1 until numFolds) {
          cuts :+= ((counts / numFolds) * i).toInt
        }
      }
      cuts :+= counts.toInt

    var currentKValues: Array[Int] = kValues

    println("^^^^^^^ scaled data partitions: " + scaledData.rdd.getNumPartitions)
    for(algorithm <- algorithms) {
      for (kValue <- currentKValues) {
        for(cutIndex<-0 until numFolds) {

          val testData = scaledData.filter(scaledData("index") < cuts(cutIndex + 1) && scaledData("index") >= cuts(cutIndex)).persist()
          val trainData = scaledData.filter(scaledData("index") >= cuts(cutIndex + 1) || scaledData("index") < cuts(cutIndex)).persist()

          trainData.show()
          println("PARTITONS " + trainData.rdd.getNumPartitions)

          val classifier = new RandomForestClassifier().setNumTrees(numRFs.toInt).
            setSeed(42L).
            setLabelCol("label").
            setFeaturesCol("features").
            setPredictionCol("prediction")

          val ppStart = System.nanoTime()
          val processedDataset = if(algorithm == "none") {
            trainData
          }
          else if(algorithm == "oversample") {
            sampleData(spark, trainData, "oversample")
          }
          else if(algorithm == "smote") {
            sampleData(spark, trainData, "smote").repartition(partitions.toInt)
          }
          else if(algorithm == "smotePlus") {
            val model = new KDTreeKNN().setFeatureCol("features")
              .setAuxCols(Array("label", "features"))
              .setLeafSize(leafSize)
                .setTreeSplits(treeSplits.toInt)
                .setTreeSplitMethod(treeSplitMethod)
            println(model.uid)

            val f = model.fit(trainData)
            f.transform(trainData).repartition(partitions.toInt)
          }
          else {
            spark.sqlContext.createDataFrame(spark.sparkContext.emptyRDD[Row], trainData.schema)
        }
        println("processedDataset for " + algorithm + " is size " + processedDataset.count() + " par: " + processedDataset.rdd.getNumPartitions)

        println("************ train data")
        val counts = getCountsByClass(spark, "label", processedDataset.toDF())
        counts.show()
        val ppEnd = System.nanoTime()
        val trainStart = System.nanoTime()

        println("************ test data")
        getCountsByClass(spark, "label", testData.toDF).show()

        val ovr = new OneVsRest().setClassifier(classifier)

        val modelRF = ovr.fit(processedDataset)
        val predictions = modelRF.transform(testData)
        trainData.printSchema()

        val result = trainData

        val combinedDf = trainData//sampleDataCluster(trainData.sparkSession, trainData)

        println("train data: " + trainData.count())
        println("new count: " + combinedDf.count())
        println("result: " + result.count())

        println("train: " + trainData.count() + " test: " + testData.count())
        predictions.show()

        getCountsByClass(spark, "label", predictions).show()

        val confusionMatrix = predictions.
          groupBy("label").
          pivot("prediction", minLabel to maxLabel).
          count().
          na.fill(0.0).
          orderBy("label")

          confusionMatrix.show()
        print("cm count: " +  confusionMatrix.count())

        val multiClassResults = calculateClassifierResults(distinctClasses, confusionMatrix)

        val predictStart = System.nanoTime()
        val predictEnd = System.nanoTime()
        val trainEnd = System.nanoTime()
        val accuracy = 0

        println("accuracy: " + accuracy)
        println("Predict Elapsed time: " + (predictEnd - trainEnd) / 1e9 + "s")
        println("Total time: " + (predictEnd - trainStart) / 1e9 + "s")

        val accTime = System.nanoTime()
        println("Acc time: " + (accTime - predictEnd) / 1e9 + "s")

        resultArray = resultArray ++ Array(Array(dataset, algorithm, dataSize, leafSize.toString, kValue.toString,
          accuracy.toString,
          "%.4f".format(multiClassResults(0).toDouble),
          "%.4f".format(multiClassResults(1).toDouble),
          "%.4f".format(multiClassResults(2).toDouble),
          "%.4f".format(multiClassResults(3).toDouble),
          "%.4f".format(multiClassResults(4).toDouble),
          "%.4f".format(multiClassResults(5).toDouble),
          "%.4f".format(multiClassResults(6).toDouble),
          "%.4f".format(multiClassResults(7).toDouble),
          "%.4f".format(multiClassResults(8).toDouble),
          "%.4f".format(multiClassResults(9).toDouble),
          ((ppEnd - ppStart) / 1e9).toString,
          ((trainEnd - trainStart) / 1e9).toString,
          ((predictEnd - trainEnd) / 1e9).toString,
          ((accTime - predictEnd) / 1e9).toString,
          ((accTime - ppStart) / 1e9).toString))
        }
      }
    }

    import spark.implicits._

    val csvResults = resultArray.map(x => x match {
      case Array(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20) => (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20)
    }).toSeq
    val c = spark.sparkContext.parallelize(csvResults).toDF
    val lookup = Map(
      "_1" -> "dataset",
      "_2" -> "algorithm",
      "_3" -> "dataSize",
      "_4" -> "leafCount",
      "_5" -> "kValue",
      "_6" -> "accuracy",
      "_7" -> "AvAcc",
      "_8" -> "MAvG",
      "_9" -> "RecM",
      "_10" -> "PrecM",
      "_11" -> "Recu",
      "_12" -> "Precu",
      "_13" -> "FbM",
      "_14" -> "Fbu",
      "_15" -> "AvFb",
      "_16" -> "CBA",
      "_17" -> "preprocessingTime",
      "_18" -> "trainTime",
      "_19" -> "predictTime",
      "_20" -> "accuracyTime",
      "_21" -> "totalTime"
    )
    val cols = c.columns.map(name => lookup.get(name) match {
      case Some(newname) => col(name).as(newname)
      case None => col(name)
    })

    val resultsDF = c.select(cols: _*)
    resultsDF.show()

    resultsDF.coalesce(1).
      write.format("com.databricks.spark.csv").
      option("header", "true").
      mode("overwrite").
      save(output_directory + "/" + dataset + "/" + dataSize + "_" + treeSplits)
    }
  }
