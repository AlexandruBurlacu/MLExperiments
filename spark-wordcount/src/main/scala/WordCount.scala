package project.wordcount.spark

import org.apache.spark.broadcast._
import org.apache.spark.rdd._


object WordCount {
  def wordCount(rdd:RDD[String],
                stopWords:Broadcast[Array[String]]):RDD[(String, Int)] = {
    val punctuationRegEx = "[_{}|<>,./\\[\\]:!?`\";()\\s+]"

    rdd.flatMap{
        /** turns the line to lower case and
          * splits it on any whitespace, and punctuation
          */
          line => line
                    .toLowerCase
                    .split(punctuationRegEx)
       }
       .filter(token => !(stopWords.value contains token)) // stopWords removal
       .map(token => (token, 1))
       .reduceByKey(_ + _)
  }
}
