# Databricks notebook source
# MAGIC %md # DSS3 Tech Sharing: MLflow
# MAGIC ### Prepared by: Yours truly, Kets

# COMMAND ----------

import mlflow
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle

mlflow.set_experiment(experiment_id=3555044431101888)

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split
from sklearn import utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import shap

# COMMAND ----------

df = pd.read_csv('/Workspace/Repos/alyssa.katelyn.castillo@aboitiz.com/dss3-mlflow-tech-sharing/data/cleaned_data.csv').iloc[:, 1:]

# COMMAND ----------

y = df['label']
X = df.drop(columns='label')

# Split out the training data
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=123)
 
# Split the remaining data equally into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=123)

# COMMAND ----------

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:,1]

# COMMAND ----------

# MAGIC %md ## Baseline model

# COMMAND ----------

with mlflow.start_run(run_name='untuned_random_forest'):
    n_estimators = 10
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=np.random.RandomState(123))
    model.fit(X_train, y_train)
   
    # predict_proba returns [prob_negative, prob_positive], so slice the output with [:, 1]
    predictions_test = model.predict_proba(X_test)[:,1]
    auc_score = roc_auc_score(y_test, predictions_test)
    mlflow.log_param('n_estimators', n_estimators)
    # Use the area under the ROC curve as a metric.
    mlflow.log_metric('auc', auc_score)
    wrappedModel = SklearnModelWrapper(model)
    # Log the model with a signature that defines the schema of the model's inputs and outputs. 
    # When the model is deployed, this signature will be used to validate inputs.
    signature = infer_signature(X_train, wrappedModel.predict(None, X_train))
    
    # MLflow contains utilities to create a conda environment used to serve models.
    # The necessary dependencies are added to a conda.yaml file which is logged along with the model.
    conda_env =  _mlflow_conda_env(
          additional_conda_deps=None,
          additional_pip_deps=['cloudpickle=={}'.format(cloudpickle.__version__), 'scikit-learn=={}'.format(sklearn.__version__)],
          additional_conda_channels=None,
      )
    mlflow.pyfunc.log_model('random_forest_model', python_model=wrappedModel, conda_env=conda_env, signature=signature)

# COMMAND ----------

# MAGIC %md ### Register the model in MLflow Model Registry

# COMMAND ----------

run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "untuned_random_forest"').iloc[0].run_id

# COMMAND ----------

run_id

# COMMAND ----------

model_name = 'atp_preds'
model_version = mlflow.register_model(f"runs:/{run_id}/random_forest_model", model_name)

# COMMAND ----------

from mlflow.tracking import MlflowClient
 
client = MlflowClient()
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage='Production',
)

# COMMAND ----------

# MAGIC %md ## Experiment with a new model

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope
from math import exp
import mlflow.xgboost
import numpy as np
import xgboost as xgb
 
search_space = {
  'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
  'learning_rate': hp.loguniform('learning_rate', -3, 0),
  'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
  'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
  'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
  'objective': 'binary:logistic',
  'seed': 123, # Set a seed for deterministic training
}
 
def train_model(params):
    # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
    mlflow.xgboost.autolog()
    with mlflow.start_run(nested=True):
        train = xgb.DMatrix(data=X_train, label=y_train)
        validation = xgb.DMatrix(data=X_val, label=y_val)
        # Pass in the validation set so xgb can track an evaluation metric. XGBoost terminates training when the evaluation metric
        # is no longer improving.
        booster = xgb.train(params=params, dtrain=train, num_boost_round=1000,\
                            evals=[(validation, 'validation')], early_stopping_rounds=50)
        validation_predictions = booster.predict(validation)
        auc_score = roc_auc_score(y_val, validation_predictions)
        mlflow.log_metric('auc', auc_score)
     
        signature = infer_signature(X_train, booster.predict(train))
        mlflow.xgboost.log_model(booster, 'model', signature=signature)
      
        # Set the loss to -1*auc_score so fmin maximizes the auc_score
        return {'status': STATUS_OK, 'loss': -1*auc_score, 'booster': booster.attributes()}
    
# Greater parallelism will lead to speedups, but a less optimal hyperparameter sweep. 
# A reasonable value for parallelism is the square root of max_evals.
spark_trials = SparkTrials(parallelism=10)
 
# Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a child run of a parent
# run called "xgboost_models" .
with mlflow.start_run(run_name='xgboost_models'):
    best_params = fmin(
        fn=train_model, 
        space=search_space, 
        algo=tpe.suggest, 
        max_evals=96,
        trials=spark_trials,
  )

# COMMAND ----------

best_run = mlflow.search_runs(order_by=['metrics.auc DESC']).iloc[0]
print(f'AUC of Best Run: {best_run["metrics.auc"]}')

# COMMAND ----------

new_model_version = mlflow.register_model(f'runs:/{best_run.run_id}/model', model_name)

# COMMAND ----------

# Archive the old model version
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage='Archived'
)
 
# Promote the new model version to Production
client.transition_model_version_stage(
  name=model_name,
  version=new_model_version.version,
  stage='Production'
)

# COMMAND ----------

# MAGIC %md ## Batch inference

# COMMAND ----------

# To simulate a new corpus of data, save the existing X_train data to a Delta table. 
# In the real world, this would be a new batch of data.
spark_df = spark.createDataFrame(X_train)
# Replace <username> with your username before running this cell.
table_path = 'dbfs:/alyssa.katelyn.castillo@aboitiz.com/delta/atp_data'
# Delete the contents of this path in case this cell has already been run
dbutils.fs.rm(table_path, True)
spark_df.write.format('delta').save(table_path)

# COMMAND ----------

import mlflow.pyfunc
apply_model_udf = mlflow.pyfunc.spark_udf(spark, f'models:/{model_name}/production')

# COMMAND ----------

# Read the "new data" from Delta
new_data = spark.read.format('delta').load(table_path)

# COMMAND ----------

display(new_data)

# COMMAND ----------

from pyspark.sql.functions import struct
 
# Apply the model to the new data
udf_inputs = struct(*(X_train.columns.tolist()))
 
new_data = new_data.withColumn(
  'prediction',
  apply_model_udf(udf_inputs)
)

# COMMAND ----------

display(new_data)
