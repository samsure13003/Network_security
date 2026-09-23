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

from sklearn.metrics import f1_score
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
    VotingClassifier,
)

RANDOM_STATE = 42
ENSEMBLE_TOP_N = 5  # how many of the best individually-tuned models go into the voting ensemble
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MLFLOW_TRACKING_URI = os.path.join(PROJECT_ROOT, "mlruns")
MLFLOW_DATABASE_PATH = os.path.join(PROJECT_ROOT, "mlflow.db")
MLFLOW_EXPERIMENT_NAME = "network-security"


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, best_model, classification_metric):
        # Explicit and deterministic: don't rely on whatever MLFLOW_TRACKING_URI
        # happens to be set in the shell/environment when this process starts.
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DATABASE_PATH.replace(os.sep, '/')}")
        if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
            mlflow.create_experiment(
                MLFLOW_EXPERIMENT_NAME,
                artifact_location=f"file:///{MLFLOW_TRACKING_URI.replace(os.sep, '/')}",
            )
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        logging.info(f"[mlflow] tracking_uri={mlflow.get_tracking_uri()} cwd={os.getcwd()}")
        with mlflow.start_run():
            mlflow.log_metric("f1_score", classification_metric.f1_score)
            mlflow.log_metric("precision", classification_metric.precision_score)
            mlflow.log_metric("recall_score", classification_metric.recall_score)
            mlflow.sklearn.log_model(
                best_model,
                name="model",
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree",
                    "sklearn.utils._bunch.Bunch",
                ],
            )

    @staticmethod
    def _safe_name(name: str) -> str:
        return name.replace(" ", "_").replace("(", "").replace(")", "")

    def train_model(self, X_train, y_train, X_test, y_test) -> ModelTrainerArtifact:
        # Model set mirrors the notebook's family coverage (linear, instance/kernel-based,
        # probabilistic, neural, tree-based) but drops xgboost / lightgbm as requested.
        # Naive Bayes uses GaussianNB (not BernoulliNB) since the incoming features are
        # already scaled/imputed continuous values from the data transformation stage,
        # not raw categorical codes.
        models = {
            "Random Forest": RandomForestClassifier(verbose=1, random_state=RANDOM_STATE),
            "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1, random_state=RANDOM_STATE),
            "Logistic Regression": LogisticRegression(verbose=1, max_iter=2000),
            "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
            "KNN": KNeighborsClassifier(),
            "SVM (RBF)": SVC(probability=True, verbose=0, random_state=RANDOM_STATE),
            "Naive Bayes": GaussianNB(),
            "MLP (Neural Net)": MLPClassifier(verbose=1, max_iter=500, early_stopping=True, random_state=RANDOM_STATE),
            "Extra Trees": ExtraTreesClassifier(verbose=1, random_state=RANDOM_STATE),
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
                "learning_rate": [0.1, 0.01, 0.001],
                "n_estimators": [64, 128, 256],
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
        model_report: dict = evaluate_models(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params
        )

        # Log every individually-tuned model's test score, best-first, so it's visible
        # in the log file which candidates were considered and how they ranked.
        ranked = sorted(model_report.items(), key=lambda kv: kv[1], reverse=True)
        logging.info("Model scores after hyperparameter tuning (best first):")
        for rank, (name, score) in enumerate(ranked, start=1):
            logging.info(f"  {rank}. {name}: f1_score={score:.4f}")

        # Soft-vote the top N already-tuned models on top of the individual results,
        # as in the notebook's ensembling step. Scored the same way (f1_score on the
        # test set) as evaluate_models so it can be compared on the same footing.
        top_n = min(ENSEMBLE_TOP_N, len(models))
        top_names = sorted(model_report, key=model_report.get, reverse=True)[:top_n]
        logging.info(
            f"Top {top_n} models selected for the soft-voting ensemble: "
            + ", ".join(f"{name} ({model_report[name]:.4f})" for name in top_names)
        )
        ensemble_estimators = [(self._safe_name(name), models[name]) for name in top_names]

        voting_clf = VotingClassifier(estimators=ensemble_estimators, voting="soft", n_jobs=-1)
        voting_clf.fit(X_train, y_train)

        ensemble_name = f"Soft Voting (top {top_n})"
        models[ensemble_name] = voting_clf
        model_report[ensemble_name] = f1_score(y_test, voting_clf.predict(X_test))
        logging.info(f"{ensemble_name}: f1_score={model_report[ensemble_name]:.4f}")

        best_model_score = max(model_report.values())
        best_model_name = max(model_report, key=model_report.get)
        best_model = models[best_model_name]
        logging.info(
            f"Best model overall (including ensemble): {best_model_name} "
            f"with test f1_score = {best_model_score:.4f}"
        )

        logging.info(
            f"Accuracy check: best_model_score={best_model_score:.4f}, "
            f"expected_accuracy={self.model_trainer_config.expected_accuracy}"
        )
        if best_model_score < self.model_trainer_config.expected_accuracy:
            # If you're not seeing the mlruns/ folder get created, check here first:
            # this raise happens BEFORE track_mlflow() is ever called, so no run is logged.
            raise NetworkSecurityException(
                Exception(
                    f"No model met the expected accuracy of "
                    f"{self.model_trainer_config.expected_accuracy}; "
                    f"best was {best_model_name} at {best_model_score:.4f}"
                ),
                sys,
            )

        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

        # Overfitting / underfitting guardrail: if train and test F1 diverge
        # too much, the model isn't trustworthy enough to ship.
        score_diff = abs(classification_train_metric.f1_score - classification_test_metric.f1_score)
        logging.info(
            f"Overfit/underfit check: score_diff={score_diff:.4f}, "
            f"threshold={self.model_trainer_config.overfitting_underfitting_threshold}"
        )
        if score_diff > self.model_trainer_config.overfitting_underfitting_threshold:
            # Same as above: this also raises BEFORE track_mlflow() runs.
            raise NetworkSecurityException(
                Exception(
                    f"Model {best_model_name} shows a train/test F1 gap of "
                    f"{score_diff:.4f}, above the allowed "
                    f"{self.model_trainer_config.overfitting_underfitting_threshold}. "
                    "Retrain with more regularization or more data."
                ),
                sys,
            )

        self.track_mlflow(best_model, classification_train_metric)
        self.track_mlflow(best_model, classification_test_metric)

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)

        # Also drop a deployment-ready copy alongside the preprocessor saved
        # by the data transformation stage.
        os.makedirs("final_model", exist_ok=True)
        save_object("final_model/model.pkl", best_model)

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric,
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