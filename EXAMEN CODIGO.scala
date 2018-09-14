//Pregunta 1
import org.apache.spark.sql.SparkSession

//parte 2
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")
//FL_insurance_sampleNetflix_2011_2016

//parte3 y 4
df.printSchema()
//punto5
df.select("Date", "Open","High","Low","Close").show()
//punto7
df.createOrReplaceTempView("HV")
df.sqlContext.sql("select * from HV").show()
//pregunta 8
df.select(max("High")).show
//pregunta 9
spark.close
println("Cierra sesion de spark en la base de datos")

//Pregunta 10
df.select(min("Volume"), max("Volume")).show()


//preguntas de abajo pregunta 11
//pregunta a
df.filter("Close > 600").count()
//pregunta b
df.filter("Date > 500").select(var_pop("Date"))
//pregunta c
df.select(corr("High","Volume")).show()
//pregunta d
df.select(max("Date")).show
//pregunta e
df.select(avg("Date")).show
