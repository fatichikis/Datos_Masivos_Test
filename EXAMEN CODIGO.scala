//Pregunta 1
import org.apache.spark.sql.SparkSession

//parte 2
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")
//FL_insurance_sampleNetflix_2011_2016

//parte 3
df.schema.fields.foreach(x => println(x))
df.columns

//parte 4
df.printSchema()

//punto5
df.select("Date", "Open","High","Low","Close").show()

//punto 6
df.describe().show()

//punto7
val df2 = df.withColumn("HV Ratio", df("High")/df("Volume"))
df2.show()

//pregunta 8
df.groupBy("Date").agg(max("High")).show

//pregunta 9
df.select("Close").describe().show()
df.select(mean("Close")).show()

println("Esta columna
  significa o se refieren a los valores con
  la que cerró la bolsa de valores de Netflix
  teniendo una media de... ")

//Pregunta 10
df.select(min("Volume"), max("Volume")).show()


//preguntas de abajo pregunta 11
//pregunta a
df.filter("Close > 600").count()

//pregunta b
val porcentage = df.filter($"High" > 500).count().toDouble / df.select("High").count() * 100

//pregunta c
df.select(corr("High","Volume")).show()

//pregunta d
df.select(max(year(df("Date")))).show()
df.groupBy(year(df("Date"))).agg(max("High")).show

//pregunta e
df.groupBy(month(df("Date"))).agg(avg("Close")).show
