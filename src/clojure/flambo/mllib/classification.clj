(ns flambo.mllib.classification
  (:import [org.apache.spark.mllib.classification
            LogisticRegressionWithLBFGS
            LogisticRegressionWithSGD
            NaiveBayes
            SVMWithSGD]
           [org.apache.spark.mllib.optimization
            L1Updater SimpleUpdater SquaredL2Updater GradientDescent LBFGS]
           [org.apache.spark.mllib.regression
            GeneralizedLinearAlgorithm]))

(defprotocol Algorithm
  (train [algo rdd]))

(defn make-updater
  [regularization-type]
  (case regularization-type
    :none (SimpleUpdater.)
    :L1   (L1Updater.)
    :L2   (SquaredL2Updater.)
    (throw (RuntimeException.
            (format "Invalid regularization type '%s'"
                    regularization-type)))))

(defn configure-generalized-linear-algorithm!
  [^GeneralizedLinearAlgorithm algo {:keys [^boolean intercept ^boolean validate-data]}]
  (when-not (nil? intercept)
    (.setIntercept algo intercept))
  (when-not (nil? validate-data)
    (.setValidateData algo validate-data))
  algo)

(defn configure-lbfgs-optimizer!
  [^LBFGS optimizer {:keys [tolerance num-iterations num-corrections
                            regularization-parameter regularization-type]}]
  (when-not (nil? tolerance)
    (.setConvergenceTol optimizer tolerance))
  (when-not (nil? num-iterations)
    (.setNumIterations optimizer num-iterations))
  (when-not (nil? num-corrections)
    (.setNumCorrections optimizer num-corrections))
  (when-not (nil? regularization-parameter)
    (.setRegParam optimizer regularization-parameter))
  (when-not (nil? regularization-type)
    (.setUpdater optimizer (make-updater regularization-type)))
  optimizer)

(defn logistic-regression-with-lbfgs
  [& {:keys [num-classes intercept validate-data tolerance
             num-iterations num-corrections regularization-parameter
             regularization-type] :as keys}]
  (let [algo (LogisticRegressionWithLBFGS.)]
    (when-not (nil? num-classes)
      (.setNumClasses algo num-classes))
    (configure-generalized-linear-algorithm! algo keys)
    (configure-lbfgs-optimizer! (.optimizer algo) keys)
    algo))

(extend-protocol Algorithm
  LogisticRegressionWithLBFGS
  (train [algo rdd] (.run algo rdd)))

(defn configure-sgd-optimizer!
  [^GradientDescent optimizer {:keys [step-size num-iterations mini-batch-fraction
                                      regularization-parameter regularization-type]}]
  (when-not (nil? step-size)
    (.setStepSize optimizer step-size))
  (when-not (nil? num-iterations)
    (.setNumIterations optimizer num-iterations))
  (when-not (nil? mini-batch-fraction)
    (.setMiniBatchFraction optimizer mini-batch-fraction))
  (when-not (nil? regularization-parameter)
    (.setRegParam optimizer regularization-parameter))
  (when-not (nil? regularization-type)
    (.setUpdater optimizer (make-updater regularization-type)))
  optimizer)

(defn logistic-regression-with-sgd
  [& {:keys [intercept validate-data
             mini-batch-fraction num-iterations step-size
             regularization-parameter regularization-type] :as keys}]
  (let [algo (LogisticRegressionWithSGD.)]
    (configure-generalized-linear-algorithm! algo keys)
    (configure-sgd-optimizer! (.optimizer algo) keys)
    algo))

(extend-protocol Algorithm
  LogisticRegressionWithSGD
  (train [algo rdd] (.run algo rdd)))

(defn naive-bayes
  [& {:keys [lambda]}]
  (let [algo (NaiveBayes.)]
    (when-not (nil? lambda)
      (.setLambda algo lambda))
    algo))

(extend-protocol Algorithm
  NaiveBayes
  (train [algo rdd] (.run algo rdd)))

(defn svm-with-sgd
  [& {:keys [intercept validate-data
             mini-batch-fraction num-iterations step-size
             regularization-parameter regularization-type] :as keys}]
  (let [algo (SVMWithSGD.)]
    (configure-generalized-linear-algorithm! algo keys)
    (configure-sgd-optimizer! (.optimizer algo) keys)
    algo))

(extend-protocol Algorithm
  SVMWithSGD
  (train [algo rdd] (.run algo (.rdd rdd))))
