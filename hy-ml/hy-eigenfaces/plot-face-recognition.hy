(setv --doc-- "
===================================================
Faces recognition example using eigenfaces and SVMs
===================================================

The dataset used in this example is a preprocessed excerpt of the
'Labeled Faces in the Wild', aka LFW-:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

.. -LFW: http://vis-www.cs.umass.edu/lfw/

Expected results for the top 5 most represented people in the dataset:

================== ============ ======= ========== =======
                   precision    recall  f1-score   support
================== ============ ======= ========== =======
     Ariel Sharon       0.67      0.92      0.77        13
     Colin Powell       0.75      0.78      0.76        60
  Donald Rumsfeld       0.78      0.67      0.72        27
    George W Bush       0.86      0.86      0.86       146
Gerhard Schroeder       0.76      0.76      0.76        25
      Hugo Chavez       0.67      0.67      0.67        15
       Tony Blair       0.81      0.69      0.75        36

      avg / total       0.80      0.80      0.80       322
================== ============ ======= ========== =======

")

(import [time [time]]
        [logging]
        [matplotlib.pyplot :as plt])

(import [sklearn.model-selection [train-test-split GridSearchCV]]
        [sklearn.datasets [fetch-lfw-people]]
        [sklearn.metrics [classification-report confusion-matrix]]
        [sklearn.decomposition [PCA]]
        [sklearn.svm [SVC]])

(print --doc--)

;; Display progress logs on stdout
(apply logging.basicConfig [] {:level logging.INFO :format "%(asctime)s %(message)s"})

;; ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Download the data, if not already on disk and load it as numpy arrays

(setv lfw-people (fetch-lfw-people :min-faces-per-person 70 :resize 0.4))

;; introspect the images arrays to find the shapes (for plotting)
(setv (, n-samples h w)  (. lfw-people images shape))

;; for machine learning we use the 2 data directly (as relative pixel
;; positions info is ignored by this model)
(setv X (. lfw-people data))
(setv n-features (. X shape [1]))

;; the label to predict is the id of the person
(setv y (. lfw-people target))
(setv target-names (. lfw-people target-names))
(setv n-classes (. target-names shape [0]))

(print "Total dataset size:")
(print "n-samples:" n-samples)
(print "n-features:" n-features)
(print "n-classes:" n-classes)

;; ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Split into a training set and a test set using a stratified k fold

;; split into a training and testing set
(setv (, X-train X-test y-train y-test)
  (apply train-test-split [X y]
    {:test-size 0.25 :random-state 42}))

;; ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
;; dataset): unsupervised feature extraction / dimensionality reduction
(setv n-components 150)

(print "Extracting the top" n-components "eigenfaces from" (. X-train shape [0]) "faces")
(setv t0 (time))
(setv pca (apply PCA [] {:n-components n-components :svd-solver "randomized" :whiten True}))
((. pca fit) X-train)
(print (.format "done in {0:.3f}s" (- (time) t0)))

(setv eigenfaces ((. pca components- reshape) (, n-components h w)))

(print "Projecting the input data on the eigenfaces orthonormal basis")
(setv t0 (time))
(setv X-train-pca ((. pca transform) X-train))
(setv X-test-pca ((. pca transform) X-test))
(print (.format "done in {0:.3f}s" (- (time) t0)))

;; ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Train a SVM classification model

(print "Fitting the classifier to the training set")
(setv t0 (time))
(setv param-grid {"C" [1e3 5e3 1e4 5e4 1e5]
                  "gamma" [0.0001 0.0005 0.001 0.005 0.01 0.1]})
(setv clf (GridSearchCV (apply SVC [] {:kernel "rbf" :class-weight "balanced"}) param-grid))
(setv clf ((. clf fit) X-train-pca y-train))
(print (.format "done in {0:.3f}s" (- (time) t0)))
(print "Best estimator found by grid search:")
(print (. clf best-estimator-))

;; ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Quantitative evaluation of the model quality on the test set

(print "Predicting people's names on the test set")
(setv t0 (time))
(setv y-pred ((. clf predict) X-test-pca))
(print (.format "done in {0:.3f}s" (- (time) t0)))

(print (apply classification-report [y-test y-pred] {:target-names target-names}))
(print (apply confusion-matrix [y-test y-pred] {:labels (range n-classes)}))


;; ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Qualitative evaluation of the predictions using matplotlib

(defn plot-gallery [images titles h w &optional [n-row 3] [n-col 4]]
    ;; Helper function to plot a gallery of portraits
    (plt.figure :figsize (, (* 1.8 n-col) (* 2.4 n-row)))

    (plt.subplots-adjust :bottom 0 :left .01 :right .99 :top .90 :hspace .35)

    (for [i (range (* n-row n-col))]
            (.subplot plt n-row n-col (+ i 1))
            (plt.imshow (.reshape (get images i) (, h w)) :cmap (. plt cm gray))
            (plt.title (get titles i) :size 12)
            (plt.xticks (,))
            (plt.yticks (,))))

;; plot the result of the prediction on a portion of the test set

(defn title [y-pred y-test target-names i]
    (setv pred-name (get (.rsplit (get target-names (get y-pred i)) " " 1) -1)
          true-name (get (.rsplit (get target-names (get y-test i)) " " 1) -1))
    (+ "predicted: " pred-name "\ntrue:     " true-name))

(setv prediction-titles (list-comp (title y-pred y-test target-names i)
                                   [i (range (get y-pred.shape 0))]))

(plot-gallery X-test prediction-titles h w)

;; plot the gallery of the most significative eigenfaces

(setv eigenface-titles (list-comp (.format "eigenface {}" i)
                                  [i (range (. eigenfaces shape [0]))]))
(plot-gallery eigenfaces eigenface-titles h w)

(.show plt)
