package project.wordcount.spark

import org.apache.spark._


object Main {
  def main(args: Array[String]) {
    
    val conf = new SparkConf().setAppName("WordCount App")
    val sc = new SparkContext(conf)

    val stopWords = sc.broadcast(StopWords.values)
    
    val iliad = sc.textFile("MLExperiments/spark-wordcount/data/iliad.txt")
    val odyssey = sc.textFile("MLExperiments/spark-wordcount/data/odyssey.txt")
    
    val iliadCounts = WordCount.wordCount(iliad, stopWords)
    val odysseyCounts = WordCount.wordCount(odyssey, stopWords)
    
    println(s"Iliad has roughly ${iliadCounts.count()} words.")
    println(s"Odyssey has roughly ${odysseyCounts.count()} words.")

    iliadCounts.saveAsTextFile("MLExperiments/tmp/wordCountsIl")
    odysseyCounts.saveAsTextFile("MLExperiments/tmp/wordCountsOd")
    
    sc.stop()
  }
}
