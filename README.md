![Flambo](http://static1.wikia.nocookie.net/__cb20120216165717/adventuretimewithfinnandjake/images/e/ee/Flambos_fire_magic.jpg)

# flambo

A Clojure DSL for Apache Spark

## Usage

In a REPL:

```clojure
(require '[flambo.conf :as conf])
(require '[flambo.api :as f])

;; make a SparkConf
(def c (-> (conf/spark-conf) (conf/master "local[2]") (conf/app-name "flambo")))

;; start a SparkContext
(def ctx (f/spark-context c))

;; make an RDD
(def xs (f/parallelize ctx (range 1000)))

;; define a serializable spark operation
(f/defsparkfn square [x] (* x x))

;; do stuff to the RDD, define and use an inline op
(-> xs (f/map square) (f/filter (f/sparkop [x] (< x 10))) f/collect)
```

## Kryo

Flambo requires spark is configured to use kryo for serialization. This is configured by default using system properties.

If you need to register custom serializers, extend `flambo.kryo.BaseFlamboRegistrator` and override it's `register` method. Finally, configure your SparkContext to use your custom registrator by setting `spark.kryo.registrator` to your custom class.

There is a convenience macro for creating registrators, `flambo.kryo.defregistrator`. The namespace where a registrator is defined should be AOT compiled.

```clojure
(require '[flambo.kryo :as kryo])

(kryo/defregistrator flameprincess [kryo]
(.register kryo FlamePrincessHeat (FlamePrincessHeatSerializer.)))

(def c (-> (conf/spark-conf) (conf/set "spark.kryo.registrator" "my.namespace.registrator.flameprincess")
```

## License

Copyright © 2014 Soren Macbeth

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
