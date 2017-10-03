(import [--future-- [print-function]])

(import [tensorflow :as tf])

;; Import MNIST dataset
(import [tensorflow.examples.tutorials.mnist [input-data]])
(setv mnist (.read-data-sets input-data "/tmp/data/" :one-hot True))

;; Network parameters
(def *learning-rate* 0.01)
(def *training-epochs* 25)
(def *batch-size* 100)
(def *display-step* 1)
(def *logs-path* "/tmp/tensorflow_logs/example/")

(def *n-hidden-1* 256)
(def *n-hidden-2* 256)
(def *n-input* 784)
(def *n-classes* 10)

(setv X (tf.placeholder tf.float32 [None 784] :name "Input")
	  y (tf.placeholder tf.float32 [None 10] :name "Labels"))

;; Layers weight & bias
(setv weights {
    "w1" (.Variable tf (.random-normal tf [*n-input* *n-hidden-1*]) :name "W1")
    "w2" (.Variable tf (.random-normal tf [*n-hidden-1* *n-hidden-2*]) :name "W2")
    "w3" (.Variable tf (.random-normal tf [*n-hidden-2* *n-classes*]) :name "W3")
}
	  biases {
    "b1" (.Variable tf (.random-normal tf [*n-hidden-1*]) :name "b1")
    "b2" (.Variable tf (.random-normal tf [*n-hidden-2*]) :name "b2")
    "b3" (.Variable tf (.random-normal tf [*n-classes*]) :name "b3")
})

;; Model
(defn multilayer-perceptron [x weights biases]
    
    (setv layer-1 (tf.add (tf.matmul x (get weights "w1")) (get biases "b1"))
    	  layer-1 ((. tf nn relu) layer-1))
    
    ((. tf summary histogram) "relu1" layer-1)
    
    (setv layer-2 (tf.add (tf.matmul layer-1 (get weights "w2")) (get biases "b2"))
    	  layer-2 ((. tf nn relu) layer-2))
    
    ((. tf summary histogram) "relu2" layer-2)
    
    (setv out_layer (tf.add (tf.matmul layer-2 (get weights "w3")) (get biases "b3")))

    out_layer)


;; Scoping TF ops
(with [(tf.name_scope "Model")]
    ;; Build model
    (setv pred (multilayer-perceptron X weights biases)))

(with [(tf.name_scope "Loss")]
    ;; Cost function
    (setv loss (tf.reduce_mean
    	((. tf nn softmax-cross-entropy-with-logits)
    	  :logits pred :labels y))))

(with [(tf.name_scope "SGD")]
    (setv optimizer ((. tf train GradientDescentOptimizer) *learning-rate*))
    
    (setv grads (tf.gradients loss (tf.trainable-variables))
    	  grads (list (zip grads (tf.trainable-variables))))

    ;; Update all variables according to their gradient
    (setv apply-grads (.apply-gradients optimizer :grads-and-vars grads)))

(with [(tf.name_scope "Accuracy")]
    (setv acc (tf.equal (tf.argmax pred 1) (tf.argmax y 1))
          acc (tf.reduce-mean (tf.cast acc tf.float32))))

;; Initializing the variables
(setv init (tf.global_variables_initializer))


((. tf summary scalar) "loss" loss)

((. tf summary scalar) "accuracy" acc)

(for [var (tf.trainable_variables)]
    ((. tf summary histogram) var.name var))

(for [[grad var] grads]
    ((. tf summary histogram) (+ var.name "/gradient") grad))

(setv merged-summary-op ((. tf summary merge_all)))

;; Training
(with [sess (tf.Session)]

    (.run sess init)

    (setv summary_writer ((. tf summary FileWriter) *logs-path*
                                  :graph (tf.get-default-graph)))

    (for [epoch (range *training-epochs*)]
        (setv avg_cost 0.
              total_batch (int (/ (. mnist train num_examples) *batch-size*)))
            
        (for [i (range total-batch)]
            (setv [batch-xs batch-ys] ((. mnist train next_batch) *batch-size*))
                       
            (setv [_ c summary] (.run sess [apply-grads loss merged-summary-op]
                                    :feed_dict {X batch-xs y batch-ys}))
                        
            (.add_summary summary_writer summary (* (+ total-batch i) epoch))
                      
            (setv avg-cost (+ avg-cost (/ c total_batch))))
    
        (if (= (% (+ 1 epoch) *display-step*) 0)
            (print (.format "Epoch: {} cost={:.9f}" (+ 1 epoch) avg-cost))))

    (print "Optimization Finished!")

    (print "Accuracy:" (acc.eval {X (. mnist test images) y (. mnist test labels)}))

    (print "Run the command line:\n"
          "--> tensorboard --logdir=/tmp/tensorflow_logs "
          "\nThen open http://0.0.0.0:6006/ into your web browser"))
