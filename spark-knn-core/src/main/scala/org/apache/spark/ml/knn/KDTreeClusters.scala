package org.apache.spark.ml.knn

import breeze.linalg._
import org.apache.spark.ml.knn.KNN._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.util.random.XORShiftRandom

import scala.collection.mutable


/**
  * A [[TreeCluster]] is used to store data points used in k-NN search. It represents
  * a binary tree node. It keeps track of the pivot vector which closely approximate
  * the center of all vectors within the node. All vectors are within the radius of
  * distance to the pivot vector. Finally it knows the number of leaves to help
  * determining partition index.
  */
private[ml] abstract class TreeCluster extends Serializable {
  val leftChild: TreeCluster
  val rightChild: TreeCluster
  val size: Int
  val leafCount: Int
  val pivot: VectorWithNorm
  val radius: Double

  def iterator: Iterator[RowWithVector]

  /**
    * k-NN query using pre-built [[Tree]]
    * @param v vector to query
    * @param k number of nearest neighbor
    * @return a list of neighbor that is nearest to the query vector
    */
  def query(v: Vector, k: Int = 1): Iterable[(RowWithVector, Double)] = query(new VectorWithNorm(v), k)
  def query(v: VectorWithNorm, k: Int): Iterable[(RowWithVector, Double)] = query(new KNNCandidatesCluster(v, k)).toIterable

  /**
    * Refine k-NN candidates using data in this [[TreeCluster]]
    */
  private[knn] def query(candidates: KNNCandidatesCluster): KNNCandidatesCluster

  /**
    * Compute QueryCost defined as || v.center - q || - r
    * when >= v.r node can be pruned
    * for MetricNode this can be used to determine which child does queryVector falls into
    */
  private[knn] def distance(candidates: KNNCandidatesCluster): Double = distance(candidates.queryVector)

  private[knn] def distance(v: VectorWithNorm): Double =
    if(pivot.vector.size > 0) pivot.fastDistance(v) else 0.0
}

private[knn]
case object EmptyCluster extends TreeCluster {
  override val leftChild = this
  override val rightChild = this
  override val size = 0
  override val leafCount = 0
  override val pivot = new VectorWithNorm(Vectors.dense(Array.empty[Double]))
  override val radius = 0.0

  override def iterator: Iterator[RowWithVector] = Iterator.empty
  override def query(candidates: KNNCandidatesCluster): KNNCandidatesCluster = candidates
}

private[knn]
case class LeafCluster (data: IndexedSeq[RowWithVector],
                 pivot: VectorWithNorm,
                 radius: Double) extends TreeCluster {
  override val leftChild = EmptyCluster
  override val rightChild = EmptyCluster
  override val size = data.size
  override val leafCount = 1

  override def iterator: Iterator[RowWithVector] = data.iterator

  // brute force k-NN search at the leaf
  override def query(candidates: KNNCandidatesCluster): KNNCandidatesCluster = {
    val sorted = data
      .map{ v => (v, candidates.queryVector.fastDistance(v.vector)) }
      .sortBy(_._2)

    for((v, d) <- sorted if candidates.notFull ||  d < candidates.maxDistance)
      candidates.insert(v, d)

    candidates
  }
}

private[knn]
object LeafCluster {
  def apply(data: IndexedSeq[RowWithVector]): LeafCluster = {
    val vectors = data.map(_.vector.vector.asBreeze)
    val (minV, maxV) = vectors.foldLeft((vectors.head, vectors.head)) {
      case ((accMin, accMax), bv) =>
        (min(accMin, bv), max(accMax, bv))
    }
    val pivot = new VectorWithNorm((minV + maxV) / 2.0)
    val radius = math.sqrt(squaredDistance(minV, maxV)) / 2.0
    LeafCluster(data, pivot, radius)
  }
}

/**
*
* KD-Tree
*/

private[knn]
case class KDTreeCluster(
                      pivot: VectorWithNorm,
                      median: Double,
                      axis: Int,
                      radius: Double,
                      leftChild: TreeCluster,
                      rightChild: TreeCluster
                 ) extends TreeCluster
  {
  override val size = leftChild.size + rightChild.size
  override val leafCount = leftChild.leafCount + rightChild.leafCount



  override def iterator: Iterator[RowWithVector] = leftChild.iterator ++ rightChild.iterator
  override def query(candidates: KNNCandidatesCluster): KNNCandidatesCluster = {
    lazy val leftQueryCost = leftChild.distance(candidates)
    lazy val rightQueryCost = rightChild.distance(candidates)

    if(candidates.notFull ||
      leftQueryCost - candidates.maxDistance < leftChild.radius ||
      rightQueryCost - candidates.maxDistance < rightChild.radius){ ///FIXME
      val remainingChild = {
        if (leftQueryCost <= rightQueryCost) {
          leftChild.query(candidates)
          rightChild
        } else {
          rightChild.query(candidates)
          leftChild
        }
      }
      // check again to see if the remaining child is still worth looking
      if (candidates.notFull ||
        remainingChild.distance(candidates) - candidates.maxDistance < remainingChild.radius) {  ///FIXME
        remainingChild.query(candidates)
      }
    }
    candidates
  }
}

object KDTreeCluster {
  /**
    * Build a (kd)[[TreeCluster]] that facilitate k-NN query
    *
    * @param data vectors that contain all training data
    * @param seed random number generator seed used in pivot point selecting
    * @return a [[TreeCluster]] can be used to do k-NN query
    */

  def build(data: IndexedSeq[RowWithVector], method: String, leafSize: Int = 1, axis: Int): Array[IndexedSeq[RowWithVector]] ={
    val size = data.size
    if (size == 0) {
      //EmptyCluster
      Array[IndexedSeq[RowWithVector]]()
    } else if (size <= leafSize) { //FIXME - add support for multiple leaf examples in leaf node, just use the pivot vector?
      Array[IndexedSeq[RowWithVector]](data)
    } else {

        def getAxisData(axisIndex: Int): (Int, Double) = {
          val values: IndexedSeq[Double] = data.map(x=>x.vector.vector).map(x=>x(axisIndex))
          val mean = values.sum/values.length
          val stdDev = Math.sqrt(values.map( _ - mean).map(t => t*t).sum/values.length)
          (axisIndex, stdDev)
        }

        def getAxisData2(axisIndex: Int): (Int, Double) = {

          val sorted: IndexedSeq[(Vector, Double)] = data.map(x => (x.vector.vector, x.vector.vector(axisIndex))).sortBy(_._2)
          val medianIndex = (sorted.length / 2.0).toInt

          val left: Array[Array[Double]] = sorted.slice(0, medianIndex.toInt).map(x => x._1.toArray).toArray
          val right = if (sorted.length < 3) IndexedSeq[RowWithVector]().toArray
          else sorted.slice(medianIndex.toInt, sorted.length).map(x => x._1).toArray

          val leftMean = left.toVector.transpose.map(_.sum / left.length).toArray
          val rightMean = left.toVector.transpose.map(_.sum / right.length).toArray

        var distance = 0.0
          for(i<-0 until leftMean.length ) {
            distance += Math.pow(leftMean(i) - rightMean(i),2)
          }
          (axisIndex, distance)
        }
        val dataLength = data(0).vector.vector.size

        val nextAxis = if(method =="minSD") {
          val bestAxis = (0 until dataLength).map(x => getAxisData(x))
          bestAxis.minBy(_._2)._1
        }
        else if(method =="maxSD") {
          val bestAxis = (0 until dataLength).map(x => getAxisData(x))
          bestAxis.maxBy(_._2)._1
        }
        else if(method == "random") {
          val r = new scala.util.Random
          r.nextInt(dataLength)
        }
        else {
          (axis + 1) % dataLength
        }

        val sorted: IndexedSeq[(RowWithVector, Double)] = data.map(x => (x, x.vector.vector(nextAxis))).sortBy(_._2)
        val medianIndex = (sorted.length / 2.0).toInt

        val left = sorted.slice(0, medianIndex.toInt).map(x => x._1)
        val right = if (sorted.length < 3) IndexedSeq[RowWithVector]()
        else sorted.slice(medianIndex.toInt, sorted.length).map(x => x._1)

        val x: Array[IndexedSeq[RowWithVector]] = build(left, method, leafSize, nextAxis)
        val y: Array[IndexedSeq[RowWithVector]] = build(right, method, leafSize, nextAxis)
        val xx: Array[IndexedSeq[RowWithVector]] = x ++ y

        xx
      }
  }
}

/**
  * Structure to maintain search progress/results for a single query vector.
  * Internally uses a PriorityQueue to maintain a max-heap to keep track of the
  * next neighbor to evict.
  *
  * @param queryVector vector being searched
  * @param k number of neighbors to return
  */
private[knn]
class KNNCandidatesCluster(val queryVector: VectorWithNorm, val k: Int) extends Serializable {
  private[knn] val candidates = mutable.PriorityQueue.empty[(RowWithVector, Double)] {
    Ordering.by(_._2)
  }

  // return the current maximum distance from neighbor to search vector
  def maxDistance: Double = if(candidates.isEmpty) 0.0 else candidates.head._2
  // insert evict neighbor if required. however it doesn't make sure the insert improves
  // search results. it is caller's responsibility to make sure either candidate list
  // is not full or the inserted neighbor brings the maxDistance down
  def insert(v: RowWithVector, d: Double): Unit = {
    while(candidates.size >= k) candidates.dequeue()
    candidates.enqueue((v, d))
  }
  def insert(v: RowWithVector): Unit = insert(v, v.vector.fastDistance(queryVector))
  def tryInsert(v: RowWithVector): Unit = {
    val distance = v.vector.fastDistance(queryVector)
    if(notFull || distance < maxDistance) insert(v, distance)
  }
  def toIterable: Iterable[(RowWithVector, Double)] = candidates
  def notFull: Boolean = candidates.size < k
}
