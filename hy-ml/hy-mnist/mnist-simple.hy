(import [tensorflow :as tf])

(import [tensorflow.examples.tutorials.mnist [input_data]])

(setv mnist (.read_data_sets input_data "MNIST_data/" :one_hot True))

(setv x (tf.placeholder tf.float32 [None 784] :name "X"))

(setv W (tf.Variable (tf.zeros [784 10]) :name "Weights")
      b (tf.Variable (tf.zeros [10]) :name "Bias"))

(setv y ((. tf nn softmax) (+ (.matmul tf x W) b))
      y-hat (tf.placeholder tf.float32 [None 10] :name "Y-hat"))

(setv cross-entropy ((. tf nn softmax_cross_entropy_with_logits)
                        :labels y-hat
                        :logits (+ (.matmul tf x W) b)))

(setv train-step (.minimize ((. tf train GradientDescentOptimizer) 0.5) cross-entropy))

(setv correct_prediction (.equal tf (.argmax tf y 1) (.argmax tf y-hat 1))
      accuracy (.reduce_mean tf (.cast tf correct_prediction tf.float32)))

(setv sess (tf.InteractiveSession))

(.run (tf.global_variables_initializer))

(setv file_writer ((. tf summary FileWriter) "./logs" sess.graph))

(for [_ (range (* 55000 3))]
    (setv (, batch-xs batch-ys) ((. mnist train next-batch) 100))
    (setv summary (.run sess (, train-step accuracy)
                   :feed-dict {"X:0" batch-xs "Y-hat:0" batch-ys})))

(print "Accuracy " (.run sess accuracy
        :feed-dict {"X:0" (. mnist test images) "Y-hat:0" (. mnist test labels)}))
