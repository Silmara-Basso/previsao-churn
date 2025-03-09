# Pipeline de Machine Learning em Cluster Spark Para Previsão de Churn - Treinamento e Deploy
# Script de Treino do Modelo

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName("Churn-Treino").getOrCreate()

df_sil = spark.read.csv("hdfs:///opt/spark/data/dataset.csv", header=True, inferSchema=True)

# Processamento dos Dados

# Convertendo colunas categóricas para representação numérica
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df_sil) for column in ["Plano", "TempoContrato"]]

# Criando o vetor de recursos
assembler = VectorAssembler(inputCols=["Idade", 
                                       "UsoMensal", 
                                       "SatisfacaoCliente", 
                                       "ValorMensal", 
                                       "Plano_index", 
                                       "TempoContrato_index"], 
                            outputCol="features")

# Construção do Modelo de Machine Learning

dados_treino, dados_teste = df_sil.randomSplit([0.7, 0.3])

modelo_rf = RandomForestClassifier(labelCol="Churn", featuresCol="features")

pipeline = Pipeline(stages=indexers + [assembler, modelo_rf])

# Treinando o modelo
modelo_sil = pipeline.fit(dados_treino)
previsoes = modelo_sil.transform(dados_teste)

# Avaliando o modelo
avaliador = BinaryClassificationEvaluator(labelCol="Churn")
acuracia = avaliador.evaluate(previsoes)

acuracia_df = spark.createDataFrame([Row(acuracia=acuracia)])

selected_columns = ["Idade", "UsoMensal", "Plano", "SatisfacaoCliente", "TempoContrato", "ValorMensal", "Churn", "prediction"]
previsoes_para_salvar = previsoes.select(selected_columns)

modelo_sil.write().overwrite().save("hdfs:///opt/spark/data/modelo")
acuracia_df.write.csv("hdfs:///opt/spark/data/acuracia", mode="overwrite")
previsoes_para_salvar.write.csv("hdfs:///opt/spark/data/previsoes", mode="overwrite")


spark.stop()
