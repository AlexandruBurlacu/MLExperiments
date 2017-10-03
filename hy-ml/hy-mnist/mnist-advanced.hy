(import [--future-- [division print_function absolute-import]])

(import [tensorflow.examples.tutorials.mnist [input-data]])
(setv mnist (.read_data_sets input_data "/tmp/data/" :one_hot False))

(import [tensorflow :as tf])

(def *learning-rate* 0.001)
(def *num-steps* 2000)
(def *batch-size* 128)


(def *num-input* 784)
(def *num-classes* 10)
(def *dropout* 0.75)

(defn conv-net [x-dict n-classes dropout reuse is-training]
    
    (with [(tf.variable_scope "ConvNet" :reuse reuse)]
    
        (setv x (get x-dict "images")
              x (tf.reshape x :shape [-1 28 28 1]))
    
        (setv conv1 ((. tf layers conv2d) x 32 5 :activation (. tf nn relu))
              conv1 ((. tf layers max_pooling2d) conv1 2 2))
    
        (setv conv2 ((. tf layers conv2d) conv1 64 3 :activation (. tf nn relu))
              conv2 ((. tf layers max_pooling2d) conv2 2 2))
    
        (setv fc1 ((. tf contrib layers flatten) conv2)
              fc1 ((. tf layers dense) fc1 1024)
          	  fc1 ((. tf layers dropout) fc1 :rate dropout :training is-training))
    
        (setv out ((. tf layers dropout) fc1 n-classes)))

    out)


(defn model-fn [features labels mode]

    (setv logits-train (conv_net features *num-classes* *dropout* :reuse False
                                :is-training True)
          logits-test (conv_net features *num-classes* *dropout* :reuse True
                               :is-training False))

    (setv pred-classes (tf.argmax logits-test :axis 1)
          pred-probas ((. tf nn softmax) logits-test))

    (if (= (mode (. tf estimator ModeKeys PREDICT)))
        ((. tf estimator EstimatorSpec) mode :predictions pred-classes))
    (else

    (setv loss-op (tf.reduce-mean
                     ((. tf nn sparse-softmax-cross-entropy-with-logits)
                     :logits logits-train :labels (tf.cast labels :dtype tf.int32))))

    (setv optimizer ((. tf train AdamOptimizer) :learning-rate *learning-rate*))
    (setv train-op (optimizer.minimize loss-op :global-step ((. tf train get-global-step))))

    (setv acc-op ((. tf metrics accuracy) :labels labels :predictions pred-classes))

    (setv estim-specs ((. tf estimator EstimatorSpec)
            :mode mode
            :predictions pred_classes
            :loss loss-op
            :train-op train-op
            :eval-metric-ops {"accuracy" acc-op}))

    estim-specs))


(setv model ((. tf estimator Estimator) model_fn))

(setv input-fn ((. tf estimator inputs numpy_input_fn)
    :x {"images" (. mnist train images)} :y (. mnist train labels)
    :batch-size *batch-size* :num-epochs None :shuffle True))

(.train model input-fn :steps *num-steps*)


(setv input-fn ((. tf estimator inputs numpy_input_fn)
    :x {"images" (. mnist test images)} :y (. mnist test labels)
    :batch-size *batch-size* :num-epochs None :shuffle False))

(setv e (model.evaluate input-fn))

(print "Testing Accuracy:" (get e "accuracy"))
