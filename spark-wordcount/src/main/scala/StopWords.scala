package project.wordcount.spark

import scala.io.Source


object StopWords {
  private val file = Source.fromFile("MLExperiments/spark-wordcount/data/stopWords.txt")

  val values = file.mkString.split(",")
}
