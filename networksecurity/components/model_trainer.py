import os
import sys

import mlflow

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models,
)
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, best_model, classificationmetric):

        mlflow.set_experiment("network-security")
        with mlflow.start_run():
                    f1_score=classificationmetric.f1_score
                    precision_score=classificationmetric.precision_score
                    recall_score=classificationmetric.recall_score
        
                    
        
                    mlflow.log_metric("f1_score",f1_score)
                    mlflow.log_metric("precision",precision_score)
                    mlflow.log_metric("recall_score",recall_score)
                    mlflow.sklearn.log_model(
                        best_model,
                        name="model",
                        skops_trusted_types=["sklearn.tree._tree.Tree"],
                    )
                    

    def train_model(self, X_train, y_train, x_test, y_test) -> ModelTrainerArtifact:
        # Model set mirrors the notebook's family coverage (linear, instance/kernel-based,
        # probabilistic, neural, tree-based) but drops xgboost / lightgbm as requested.
        # Naive Bayes uses GaussianNB (not BernoulliNB) since the incoming features are
        # already scaled/imputed continuous values from the data transformation stage,
        # not raw categorical codes.
        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Logistic Regression": LogisticRegression(verbose=1, max_iter=2000),
            "AdaBoost": AdaBoostClassifier(),
            "KNN": KNeighborsClassifier(),
            "SVM (RBF)": SVC(probability=True, verbose=0),
            "Naive Bayes": GaussianNB(),
            "MLP (Neural Net)": MLPClassifier(verbose=1, max_iter=500, early_stopping=True),
            "Extra Trees": ExtraTreesClassifier(verbose=1),
        }

        # evaluate_models() only runs GridSearchCV (cv=3), so every grid here is a small,
        # exhaustive set of discrete values rather than the wider random-search
        # distributions the notebook used for the heavier models.
        params = {
            "Decision Tree": {
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": [None, 5, 10, 20],
            },
            "Random Forest": {
                "n_estimators": [64, 128, 256],
                "max_depth": [None, 10, 20],
            },
            "Gradient Boosting": {
                "learning_rate": [0.1, 0.05, 0.01],
                "subsample": [0.7, 0.8, 0.9],
                "n_estimators": [64, 128, 256],
            },
            "Logistic Regression": {
                "C": [0.1, 1.0, 10.0],
            },
            "AdaBoost": {
                "learning_rate": [0.1, 0.001],
                "n_estimators": [64, 256],
            },
            "KNN": {
                "n_neighbors": [5, 7, 11, 15],
                "weights": ["uniform", "distance"],
            },
            "SVM (RBF)": {
                "C": [0.1, 1, 10],
                "gamma": ["scale", 0.01, 0.1],
            },
            "Naive Bayes": {
                "var_smoothing": [1e-9, 1e-8, 1e-7],
            },
            "MLP (Neural Net)": {
                "hidden_layer_sizes": [(64,), (128,), (64, 32)],
                "alpha": [0.0001, 0.001],
            },
            "Extra Trees": {
                "n_estimators": [64, 128, 256],
                "max_depth": [None, 10, 20],
            },
        }

        # evaluate_models fits each model with its GridSearchCV best params in place
        # (model.set_params(...); model.fit(...)) before scoring, so `models[name]`
        # is already a tuned, fitted estimator once this call returns.
        model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                                  models=models,param=params)
                
        ## To get best model score from dict
        best_model_score = max(sorted(model_report.values()))
        
        ## To get best model name from dict
        
        best_model_name = list(model_report.keys())[
                    list(model_report.values()).index(best_model_score)
                ]
        best_model = models[best_model_name]
        y_train_pred=best_model.predict(X_train)
        
        classification_train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)
                
                ## Track the experiements with mlflow
        self.track_mlflow(best_model,classification_train_metric)
        
        
        y_test_pred=best_model.predict(x_test)
        classification_test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)
        
        self.track_mlflow(best_model,classification_test_metric)
        
        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
                    
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)
        
        Network_Model=NetworkModel(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=NetworkModel)
                #model pusher
        save_object("final_model/model.pkl",best_model)
                
        
                ## Model Trainer Artifact
        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                     train_metric_artifact=classification_train_metric,
                                     test_metric_artifact=classification_test_metric
                                     )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            return self.train_model(X_train, y_train, X_test, y_test)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e