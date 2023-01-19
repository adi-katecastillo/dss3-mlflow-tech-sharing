# Databricks notebook source
# MAGIC %md ## Experiments and Runs

# COMMAND ----------

mlflow.create_experiment() # creates a new experiment

# COMMAND ----------

mlflow.set_experiment(experiment_id='') # set the workspace experiment

# COMMAND ----------

mlflow.start_run(run_name='') # returns or starts new run

# COMMAND ----------

# MAGIC %md ## Model Registry

# COMMAND ----------

mlflow.register_model(model_uri='', name='') # registers model in model registry

# COMMAND ----------

client = MlflowClient()

# deploys model to production
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage='Production',
)

# archives a model
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage='Archive',
)

# COMMAND ----------

client = MlflowClient()

# delete a model version
client.delete_model_version(name='', version=)

# delete a registered model along with all its versions
client.delete_registered_model(name=)
