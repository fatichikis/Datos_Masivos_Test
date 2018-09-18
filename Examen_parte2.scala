//Pregunta 1 (Comienza un simple Spark Session)
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()

//parte 2 (Cargue el archivo netflix)
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")
//FL_insurance_sampleNetflix_2011_2016

//parte 3 (Cuales son los nombres de las columnas)
df.schema.fields.foreach(x => println(x)) //Imprime nombres y tipo de dato
df.columns //imprime solo los nombres
 
//parte 4 (Como es el esquema)
df.printSchema()

//punto5 (Imprime las primeras 5 columnas)
df.select("Date", "Open","High","Low","Close").show()
df.columns.take(5)

//punto 6 (usa describe() para aprender sobre el dataframe)
df.describe().show()

//punto7 (crea un dataframe con una columna llamada HV RATIO que es la relacion de HIGH frente al VOLUMEN)
val df2 = df.withColumn("HV Ratio", df("High")/df("Volume"))
df2.show()

//pregunta 8 (Que dia tuvo peak High en Price)
df.groupBy("Date").agg(max("High")).show

//pregunta 9 (Cual es el significado de la columna cerrar (close))
df.select("Close").describe().show()

println("Esta columna
  significa o se refieren a los valores con
  la que cerró la bolsa de valores de Netflix
  teniendo una media de... ")

//Pregunta 10 (Cual es el maximo y mino de la column VOLUME)
df.select(min("Volume"), max("Volume")).show()


//preguntas de abajo pregunta 11
//pregunta a (Cuantos dias fue el cierre inferior a 600)
df.filter("Close < 600").count()

//pregunta b (Que porcentaje del tiempo fue el alto mayor de 500)
val porcentage = df.filter("High > 500").count().toDouble / df.select("High").count() * 100

//pregunta c (Cual es la correlacion de person entre alto y volumen)
df.select(corr("High","Volume")).show()

//pregunta d (Cual es el maximo alto por año)
df.groupBy(year(df("Date"))).agg(max("High")).show

//pregunta e (Cual es el promedio de cierre para cada mes del calendario)
df.groupBy(month(df("Date"))).agg(avg("Close")).show

