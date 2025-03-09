# previsao-churn
Pipeline de Machine Learning em cluster Spark para previsão de Churn com treinamento e deploy com algoritmo RandomForest o algoritimo 

## O modelo foi desenvolvido usando o clone do repositório hdfs-spark
https://github.com/Silmara-Basso/hdfs-spark

`docker compose -f docker-compose.yml up -d --scale spark-worker-yarn=3`

`docker compose logs`


### Treinar o modelo e salvar o modelo

`docker exec sil-spark-master-yarn spark-submit --master yarn --deploy-mode cluster ./apps/churn-treino.py`


### Carregar o modelo treinado e fazer previsões

`docker exec sil-spark-master-yarn spark-submit --master yarn --deploy-mode cluster ./apps/churn-deploy.py`