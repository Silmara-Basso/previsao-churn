# Pipeline de Machine Learning em Cluster Spark Para Previsão de Churn - Treinamento e Deploy
# Script de Deploy do Modelo

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

spark = SparkSession.builder.appName("churn-Deploy").getOrCreate()

# Carregando o modelo do HDFS 
modelo_sil = PipelineModel.load("hdfs:///opt/spark/data/modelo")

novos_dados = spark.read.csv("hdfs:///opt/spark/data/novosdados.csv", header=True, inferSchema=True)

# Aplicando o modelo aos novos dados para fazer previsões
previsoes = modelo_sil.transform(novos_dados)
selected_columns = ["Idade", "UsoMensal", "Plano", "SatisfacaoCliente", "TempoContrato", "ValorMensal", "prediction"]
previsoes_para_salvar = previsoes.select(selected_columns)

previsoes_para_salvar.write.csv("hdfs:///opt/spark/data/previsoesnovosdados", mode="overwrite")

spark.stop()
