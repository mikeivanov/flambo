(ns flambo.mllib.regression
  (:import [org.apache.spark.mllib.regression LabeledPoint]
           [org.apache.spark.mllib.linalg Vector]))

(defn labeled-point
  [^double label ^Vector features]
  (LabeledPoint. label features))
