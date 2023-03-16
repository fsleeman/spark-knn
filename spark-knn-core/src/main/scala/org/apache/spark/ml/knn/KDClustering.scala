package org.apache.spark.ml.knn

import org.apache.log4j._
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.collection.parallel.immutable.ParSeq
import scala.util.Random


object Classifier {

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

      //FIXME - what to do if col/row sum are zero?
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
    MAvG = {  val result = Math.pow(MAvG, 1/numberOfClasses.toDouble); if(result.isNaN) 0.0 else result } //Math.pow((MAvG), (1/numberOfClasses.toDouble))
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
    val randomInts: Random = new scala.util.Random(42L)
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
      filteredDF2//.cache() --FIXME cache removed, might be required for not running out of memory
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

      val myDFs: Array[(Int, DataFrame)] = presentClasses.map(x=>(x, df.filter(df("label") === x).toDF()))
      val dfs = presentClasses.map(x => sampleDataParallel(spark, myDFs.filter(y=>y._1 == x)(0)._2, x, samplingMethod, underSampleCount, overSampleCount, smoteSampleCount))

      println("Final count ")

      val all = dfs.reduce(_ union  _)
      println("^^^^^^^^^^^^")

      all
    }

  def YYY(cls: Int, currentCount: Long, clusterIds: Seq[Int], clusteredData: Map[Int, Seq[(Int, Int, Int, Int, DenseVector)]], clusterCount: Int, randomInts: Random): Row = {
    val clusterIndex = clusterIds(randomInts.nextInt(clusterIds.length))
    val sampled: Array[(Int, Int, Int, Int, DenseVector)] = Array.fill(5)(clusteredData(clusterIndex)(randomInts.nextInt(clusteredData(clusterIndex).length)))   ///FIXME - this is the change

    //FIXME - can we dump the index column?
    val values: Array[Array[Double]] = sampled.map(x=>x._5.asInstanceOf[DenseVector].toArray)

    val ddd: Array[Double] = values.transpose.map(_.sum /values.length)
    val r2 = Row(0, cls, 0,  Vectors.dense(ddd.map(_.toDouble)))

    //FIXME - convert this to DenseVector
    r2
  }

  //FIXME - cutoff does not seem to be used
  def sampleDataSmotePlus(spark: SparkSession, df: DataFrame, samplingMethod: String, clusterKValue: Int, clusterResults: Map[Int,DataFrame], cutoff: Double=0.0): DataFrame = {
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
      val currentCase = df.filter(df("label") === l).toDF() ///FIXME - this may already been calculated
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

      val predictionsCollected = predictions.collect().map(x=>(x(0).toString.toInt, x(1).toString.toInt, x(2).toString.toInt, x(3).toString.toInt, x(4).asInstanceOf[DenseVector])).toSeq
      val clusteredData: Map[Int, Seq[(Int, Int, Int, Int, DenseVector)]] = predictionsCollected.groupBy {_._1}
      val clusterIds = clusteredData.keySet.toSeq  //.map(x=>x._1).toSeq

      println("samplesToAdd: "  + samplesToAdd)

      val t01 = System.nanoTime()

      val XXX: ParSeq[Row] = (1 to samplesToAdd.toInt).par.map { _ => YYY(l: Int, currentCount: Long, clusterIds: Seq[Int], clusteredData: Map[Int, Seq[(Int, Int, Int, Int, DenseVector)]], clusterCount: Int, randomInts: Random) }// .reduce(_ ++ _)
      samples = samples ++ XXX
      val t11 = System.nanoTime()
      println("--------- LOOP TIME: " + (t11 - t01) / 1e9 + "s")

    }
    else {
      // we already have enough samples, skip
    }

    val tX = System.nanoTime()
    val currentArray = df.rdd.map(x=>Row(x(1), x(2), x(3), x(4).asInstanceOf[DenseVector])).collect()

    samples = samples ++ currentArray

    val foo = samples.map(x=>x.toSeq).map(x=>(x(0).toString.toInt, x(1).toString.toInt, x(2).toString.toInt, x(3).asInstanceOf[DenseVector]))//asInstanceOf[mutable.WrappedArray[Double]]))

    val bar = spark.createDataFrame(spark.sparkContext.parallelize(foo))  //   createDataFrame(foo)  //spark.sparkContext.parallelize(foo).toDF()

    val bar2 = bar.withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "minorityType")
      .withColumnRenamed("_4", "features")

    val tY = System.nanoTime()
    println("--------- Combine Time: " + (tY - tX) / 1e9 + "s")
    bar2
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val input_directory = args(0)
    val output_directory = args(1)

    val dataset = args(2)
    val dataSize = args(3)
    val algorithm = args(4)

    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._


    /// FIXME - doesnt seem to work if array size is > 1
    val leafSizes = Array(100)
    val kValues = Array(5)
    val algorithms = if(algorithm == "kdtree") {
      Array("kdtree")
    }
    else if(algorithm == "hybrid") {
      Array("hyrbid")
    }
    else {
      Array("hyrbid", "kdtree")
    }

    val isAllNN = false
    var resultArray: Array[Array[String]] = Array()

    val df = spark.read.
    option("inferSchema", true).
    option("header", true).
    csv(input_directory + "/" + dataset + "/" + dataset + dataSize + ".csv")

    val dfCount = df.count()

    println("df count: " + dfCount)

    val train_index = df.rdd.zipWithIndex().map({ case (x, y) => (y, x) })

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
    val scaledData: DataFrame = if (enableDataScaling) {
      val scaler = new MinMaxScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeatures")
      val scalerModel = scaler.fit(converted)
      val scaledData: DataFrame = scalerModel.transform(converted)
      scaledData.drop("features").withColumnRenamed("scaledFeatures", "features")
    } else {
      converted
    }.cache()

    val distinctClasses = scaledData.select("label").distinct()
    val maxLabel: Int = distinctClasses.collect().map(x => x.toSeq.last.toString.toDouble.toInt).max
    val minLabel: Int = distinctClasses.collect().map(x => x.toSeq.last.toString.toDouble.toInt).min

    val numSplits = 1
      val counts = scaledData.count()
      var cuts = Array[Int]()

      cuts :+= 0
      if (numSplits < 2) {
        cuts :+= (counts * 0.2).toInt
      }
      else {
        for (i <- 1 until numSplits) {
          cuts :+= ((counts / numSplits) * i).toInt
        }
      }
      cuts :+= counts.toInt

    val currentKValues: Array[Int] = kValues

    println("^^^^^^^ scaled data partitions: " + scaledData.rdd.getNumPartitions)
    for(algorithm <- algorithms) {
      for (kValue <- currentKValues) {
        for (leafSize <- leafSizes) {

          for(cutIndex<-0 until numSplits) {
            val trainStart = System.nanoTime()

            val testData = scaledData.filter(scaledData("index") < cuts(cutIndex + 1) && scaledData("index") >= cuts(cutIndex)).persist()
            val trainDataOriginal = scaledData.filter(scaledData("index") >= cuts(cutIndex + 1) || scaledData("index") < cuts(cutIndex)).persist()
            val trainData = sampleData(spark, trainDataOriginal, "smote")
            println("oversampled: " + trainData.count())

            println("train: " + trainData.count() + " test: " + testData.count())

            val model = new KNN().setFeaturesCol("features")
              .setTopTreeSize(trainData.count().toInt / 10)
              .setTopTreeLeafSize(leafSize)
              .setSubTreeLeafSize(leafSize)
              .setSeed(42L)
              .setAuxCols(Array("label", "features"))
              .setAlgorithm(algorithm)
            if(isAllNN) {
              model.setK(kValue + 1)
            }
            else {
              model.setK(kValue)
            }

            trainData.show()
            trainData.printSchema()

            val f = model.fit(trainData)
            println(model.uid)

            val trainEnd = System.nanoTime()
            println("Train Elapsed time: " + (trainEnd - trainStart) / 1e9 + "s")


            val result: DataFrame = f.transform(testData)
            result.printSchema()
            val collected: DataFrame = result.select("index", "label", "neighbors")

            val predictions = if(isAllNN) {
              collected.map(x => (x(1).toString.toInt, x(2).asInstanceOf[mutable.WrappedArray[Any]].tail.map(y => formatResults(y.toString)))).map(z => (z._1, getPredictedLabel(z._2.toArray)))
            }
            else {
              collected.map(x => (x(1).toString.toInt, x(2).asInstanceOf[mutable.WrappedArray[Any]].map(y => formatResults(y.toString)))).map(z => (z._1, getPredictedLabel(z._2.toArray)))
            }

            val predictEnd = System.nanoTime()
            val accuracy = predictions.map(x => isCorrectPrediction(x)).collect().sum / predictions.count().toDouble
            predictions.show()

            val confusionMatrix = predictions.
              groupBy("_1").
              pivot("_2", minLabel to maxLabel).
              count().
              na.fill(0.0).
              orderBy("_1")
            confusionMatrix.show()

            val multiClassResults = calculateClassifierResults(distinctClasses, confusionMatrix)

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
              ((trainEnd - trainStart) / 1e9).toString,
              ((predictEnd - trainEnd) / 1e9).toString,
              ((accTime - predictEnd) / 1e9).toString,
              ((accTime - trainStart) / 1e9).toString))
          }
        }
      }
    }
    import spark.implicits._

    val csvResults = resultArray.map(x => x match {
        case Array(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19) => (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19)
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
      "_17" -> "trainTime",
      "_18" -> "predictTime",
      "_19" -> "accuracyTime",
      "_20" -> "totalTime"
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
      save(output_directory)
    }
  }
