from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans, DBSCAN

import pandas as pd


class ModelFactory:
    """
    ModelFactory Class

    The single source of truth for "which models exist and how do I
    create one". Centralizing model creation here means every other
    module (ModelTrainer, MLPipeline) never has to import a specific
    scikit-learn estimator directly — they just ask the factory for a
    model by name or by task type.

    Supports three task families, matching the class diagram:
        - classification : RandomForest, LogisticRegression, SVM, KNN, DecisionTree
        - regression      : RandomForest, LinearRegression, SVM, KNN, DecisionTree
        - clustering       : KMeans, DBSCAN

    Design notes:
        - recommend_models() is what powers "User Mode" in the broader
          MLPipeline: a non-technical user picks a target column, and
          the system suggests sensible models WITHOUT the user needing
          to know what "SVM" or "DBSCAN" even means.
        - create_<algorithm>(**params) methods all forward **params
          straight to the underlying scikit-learn estimator constructor,
          so any valid sklearn hyperparameter works out of the box
          without this class needing to know about every one of them.

    Attributes:
        task_type (str): "classification" | "regression" | "clustering".
        models (dict)   : Registry of currently created model instances,
                           keyed by a user-given name — lets a session
                           track multiple trained candidates at once
                           (useful for "compare models" features).

    Methods:
        recommend_models(task_type, data)     : Suggest suitable model names
                                                  for a task, with reasoning.
        get_model(model_name, **params)        : Generic factory — create
                                                  any supported model by name.
        available_models()                     : List all supported model names
                                                  grouped by task type.
        create_random_forest(**params)
        create_logistic_regression(**params)
        create_linear_regression(**params)
        create_svm(**params)
        create_knn(**params)
        create_decision_tree(**params)
        create_kmeans(**params)
        create_dbscan(**params)
    """

    # Registry mapping model_name -> (task_types it supports, constructor)
    _REGISTRY = {
        "random_forest": {
            "classification": RandomForestClassifier,
            "regression": RandomForestRegressor,
        },
        "logistic_regression": {"classification": LogisticRegression},
        "linear_regression": {"regression": LinearRegression},
        "svm": {"classification": SVC, "regression": SVR},
        "knn": {"classification": KNeighborsClassifier, "regression": KNeighborsRegressor},
        "decision_tree": {
            "classification": DecisionTreeClassifier,
            "regression": DecisionTreeRegressor,
        },
        "kmeans": {"clustering": KMeans},
        "dbscan": {"clustering": DBSCAN},
    }

    def __init__(self, task_type: str = "classification"):
        if task_type not in ("classification", "regression", "clustering"):
            raise ValueError(
                "task_type must be one of: 'classification', 'regression', 'clustering'"
            )
        self.task_type = task_type
        self.models = {}

    # ── Model recommendation (powers "User Mode") ───────────────────────
    def recommend_models(self, task_type: str, data: pd.DataFrame = None) -> list:
        """
        Suggest a shortlist of suitable models for the given task type,
        with a short plain-language reason for each — this is what lets
        a non-technical "User Mode" user get a sensible starting point
        without understanding the underlying algorithms.

        Args:
            task_type (str): "classification", "regression" or "clustering".
            data (pd.DataFrame, optional): The dataset to recommend for.
                When provided, the dataset size is used to bias the
                recommendation (e.g. avoid KNN/SVM as the top pick on
                very large datasets, since both scale poorly).

        Returns:
            list[dict]: Each item is
                {"model_name": str, "reason": str, "recommended_rank": int}
                sorted with the most recommended model first.
        """
        if task_type not in ("classification", "regression", "clustering"):
            raise ValueError(
                "task_type must be one of: 'classification', 'regression', 'clustering'"
            )

        n_rows = len(data) if data is not None else None
        large_dataset = n_rows is not None and n_rows > 20000

        if task_type == "classification":
            recs = [
                ("random_forest", "Strong all-around baseline; handles non-linear patterns and mixed feature types well."),
                ("logistic_regression", "Fast, interpretable, great baseline for linearly separable classes."),
                ("svm", "Effective on smaller, high-dimensional datasets with clear margins between classes."),
                ("knn", "Simple, no training phase; works well when similar rows tend to share a label."),
                ("decision_tree", "Highly interpretable single tree, useful for explaining individual decisions."),
            ]
        elif task_type == "regression":
            recs = [
                ("random_forest", "Robust baseline that captures non-linear relationships without heavy tuning."),
                ("linear_regression", "Fast, interpretable, ideal when the relationship between features and target is roughly linear."),
                ("svm", "Can model complex relationships on smaller datasets via different kernels."),
                ("knn", "Predicts using the average of nearby points; simple and effective for smooth target surfaces."),
                ("decision_tree", "Interpretable, captures non-linear splits, good for explaining predictions."),
            ]
        else:  # clustering
            recs = [
                ("kmeans", "Fast and effective when clusters are roughly round and similarly sized."),
                ("dbscan", "Better when clusters have irregular shapes or you need automatic outlier/noise detection."),
            ]

        # De-prioritize distance-based models (svm, knn) on large datasets,
        # since they scale poorly with dataset size.
        if large_dataset:
            recs = sorted(
                recs, key=lambda item: item[0] in ("svm", "knn")
            )

        return [
            {"model_name": name, "reason": reason, "recommended_rank": idx + 1}
            for idx, (name, reason) in enumerate(recs)
        ]

    # ── Generic factory method ───────────────────────────────────────────
    def get_model(self, model_name: str, **params):
        """
        Generic factory: create any supported model by its registry name
        for the factory's current task_type.

        Args:
            model_name (str): One of: "random_forest", "logistic_regression",
                "linear_regression", "svm", "knn", "decision_tree",
                "kmeans", "dbscan".
            **params: Hyperparameters forwarded directly to the underlying
                scikit-learn estimator constructor.

        Returns:
            object: An (untrained) scikit-learn estimator instance.
        """
        if model_name not in self._REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available models: {list(self._REGISTRY.keys())}"
            )

        task_map = self._REGISTRY[model_name]
        if self.task_type not in task_map:
            raise ValueError(
                f"Model '{model_name}' does not support task_type "
                f"'{self.task_type}'. It supports: {list(task_map.keys())}"
            )

        model = task_map[self.task_type](**params)
        self.models[model_name] = model
        return model

    # ── Discover what's available ───────────────────────────────────────
    def available_models(self) -> dict:
        """
        List every supported model name, grouped by the task type(s)
        each one supports.

        Returns:
            dict: {
                "classification": [...],
                "regression": [...],
                "clustering": [...]
            }
        """
        grouped = {"classification": [], "regression": [], "clustering": []}
        for model_name, task_map in self._REGISTRY.items():
            for task in task_map:
                grouped[task].append(model_name)
        return grouped

    # ── Named convenience constructors (matches the class diagram) ─────
    def create_random_forest(self, **params):
        """Create a RandomForest model (classifier or regressor, based on task_type)."""
        return self.get_model("random_forest", **params)

    def create_logistic_regression(self, **params):
        """Create a LogisticRegression model (classification only)."""
        return self.get_model("logistic_regression", **params)

    def create_linear_regression(self, **params):
        """Create a LinearRegression model (regression only)."""
        return self.get_model("linear_regression", **params)

    def create_svm(self, **params):
        """Create an SVM model (SVC for classification, SVR for regression)."""
        return self.get_model("svm", **params)

    def create_knn(self, **params):
        """Create a K-Nearest-Neighbors model (classifier or regressor)."""
        return self.get_model("knn", **params)

    def create_decision_tree(self, **params):
        """Create a DecisionTree model (classifier or regressor)."""
        return self.get_model("decision_tree", **params)

    def create_kmeans(self, **params):
        """Create a KMeans clustering model."""
        return self.get_model("kmeans", **params)

    def create_dbscan(self, **params):
        """Create a DBSCAN clustering model."""
        return self.get_model("dbscan", **params)
