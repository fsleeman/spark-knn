package org.apache.spark.ml.knn

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{ObjectType, StructField, _}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.knn.KNN.{KNNPartitioner, RowWithVector, VectorWithNorm}
import KDClustering.getCountsByClass

import scala.util.Random

trait KDTreeKNNParams extends Params {
  final val inputCols= new Param[Array[String]](this, "inputCols", "The input columns")
  final val featuresCol= new Param[String](this, "featuresCol", "The features column")
  final val outputCol = new Param[String](this, "outputCol", "The output column")
  final val kValue = new Param[Int](this, "kValue", "Value of k-neighbors")
  final val leafSize = new Param[Int](this, "leafSize", "Number of leaf examples in leaf nodes")
  final val treeSplits = new Param[Int](this, "treeSplits", "Number of leaf examples in leaf nodes")
  final val treeSplitMethod = new Param[String](this, "treeSplitMethod", "Number of leaf examples in leaf nodes")
  final val partitions = new Param[Int](this, "partitions", "Number of leaf examples in leaf nodes")
  final val topTreeLeafSize = new Param[Int](this, "topTreeLeafSize", "Number of leaf examples in leaf nodes")
  final val number = new Param[Int](parent = this, "number", doc="foo")
}

class KDTreeKNN(override val uid: String) extends Estimator[KDTreeModel] with KDTreeKNNParams {
  def setAuxCols(value: Array[String]) = set(inputCols, value)
  def setFeatureCol(value: String) = set(featuresCol, value)
  def setOutputCol(value: String) = set(outputCol, value)
  def setK(value: Int) = set(kValue, value)
  def setLeafSize(value: Int) = set(leafSize, value)

  def setTreeSplits(value: Int) = set(treeSplits, value)
  def setTreeSplitMethod(value: String) = set(treeSplitMethod, value)

  def setPartitions(value: Int) = set(partitions, value)

  def setNumner(value: Int) = set(number, value)

  /** @group setParam */
  def this() = this(Identifiable.randomUID("kdtree"))

  override def copy(extra: ParamMap): KDTreeKNN = {
    defaultCopy(extra)
  }

  override def transformSchema(schema: StructType): StructType = {
    // Check that the input type is a string
    val idx = schema.fieldIndex($(featuresCol))
    val field = schema.fields(idx)
    if (field.dataType != StringType) {
      throw new Exception(s"Input type ${field.dataType} did not match input type StringType")
    }
    // Add the return field
    schema.add(StructField($(outputCol), IntegerType, false))
  }

  def createTree(clsData : (Int, DataFrame)): (Int, Array[IndexedSeq[RowWithVector]]) = {
    val data: RDD[KNN.RowWithVector] = clsData._2.selectExpr($(featuresCol), $(inputCols).mkString("struct(", ",", ")"))
      .rdd
      .map(row => new KNN.RowWithVector(row.getAs[Vector](0), row.getStruct(1)))

    val sampled = data.collect().toIndexedSeq

    val leafSize = Math.max(64, sampled.size / $(treeSplits))
    println("IN " + clsData._1 + " " + sampled.length + " + leaf " + leafSize)
    val myTree: Array[IndexedSeq[RowWithVector]] = KDTreeCluster.build(sampled, $(treeSplitMethod), leafSize, axis= 0)

    println("cluster count: " + myTree.length)

    (clsData._1, myTree)
  }

  override def fit(dataset: Dataset[_]): KDTreeModel = {
    val distinctClasses = dataset.select("label").distinct()
    val presentClasses: Array[Int] = distinctClasses.select("label").rdd.map(r => r(0)).collect().map(x=>x.toString.toInt).filter(x=>x!=2)

    val datasetRepartitioned = dataset//.repartition(8)
    val clusterResults: Map[Int, DataFrame] = presentClasses.map(x=>(x, datasetRepartitioned.filter(datasetRepartitioned("label")===x).toDF)).toMap
    println("count of clusters: " + clusterResults.size)

    // class/cluster/elements
    val trees: Array[(Int, Array[IndexedSeq[RowWithVector]])] = clusterResults.map(x=>createTree(x)).toArray

    println("length of clusters: " + trees.length)
    val model = new KDTreeModel(uid, trees)
    copyValues(model)
  }
}

class KDTreeModel(override val uid: String, trees: (Array[(Int, Array[IndexedSeq[RowWithVector]])])) extends Model[KDTreeModel] with KDTreeKNNParams {
  override def copy(extra: ParamMap): KDTreeModel = {
    defaultCopy(extra)
  }
  def getK(): String ={ kValue.toString() }

  override def transformSchema(schema: StructType): StructType = {
    // Check that the input type is a string
    val idx = schema.fieldIndex($(featuresCol))
    val field = schema.fields(idx)
    if (field.dataType != StringType) {
      throw new Exception(s"Input type ${field.dataType} did not match input type StringType")
    }
    // Add the return field
    schema.add(StructField($(outputCol), IntegerType, false))
  }

  def createNewSample(tree: (Int, Array[IndexedSeq[RowWithVector]]), randomInts: Random): Row = {

    val cls = tree._1

    val currentCluster = tree._2(randomInts.nextInt(tree._2.length))
    def r = randomInts.nextInt(currentCluster.length)

    val sampled = Array(currentCluster(r), currentCluster(r), currentCluster(r), currentCluster(r), currentCluster(r))
    val values: Array[Array[Double]] = sampled.map(x=>x.vector.vector.toArray)
    val tp: Array[Double] = values.transpose.map(_.sum /values.length)

    Row(0L, cls, Vectors.dense(tp))
  }

  override def transform(dataset: Dataset[_]): DataFrame = {

    val spark = dataset.sparkSession
    import spark.implicits._

    val counts = getCountsByClass(spark, "label", dataset.toDF())

    val maxClassCount = counts.select("_2").agg(max("_2")).take(1)(0)(0).toString.toInt
    println(maxClassCount)

    val samplesToAdd = counts.map(x=>(x(0).toString.toInt, maxClassCount - x(1).toString.toInt)).collect().toMap

    val randomInts: Random = new scala.util.Random(42L)

    val result: DataFrame = trees.map(tree=>{

    println(tree._1, samplesToAdd(tree._1))
    println("~~~~~~~~~cls: " + tree._1 + " clusters: " + tree._2.length)
    val samples = (0 until samplesToAdd(tree._1)).map(x=>createNewSample(tree, randomInts))

    val mappedResults: RDD[Row] = spark.sparkContext.parallelize(samples)
    val mappedDF = spark.sqlContext.createDataFrame(mappedResults, dataset.toDF.schema)
    mappedDF

    }).reduce(_ union _).union(dataset.toDF()).orderBy(rand())

    println("------------ sampled counts: "  + result.count())
    result
  }
}
