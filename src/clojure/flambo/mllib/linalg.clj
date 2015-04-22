(ns flambo.mllib.linalg
  (:import [org.apache.spark.mllib.linalg Vector Vectors]))

(defn dense-vector
  [elements]
  (Vectors/dense (double-array elements)))

(defn sparse-vector
  ([size] (sparse-vector size []))
  ([size pairs] (apply sparse-vector size (apply map vector pairs)))
  ([size indices values] (Vectors/sparse size
                                         (int-array indices)
                                         (double-array values))))

(defn unpack [^Vector vector]
  (into [] (.toArray vector)))

