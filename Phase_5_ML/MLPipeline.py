import pandas as pd
from sklearn.model_selection import train_test_split

from Phase_5_ML.ModelFactory import ModelFactory
from Phase_5_ML.ModelTrainer import ModelTrainer
from Phase_5_ML.ModelEvaluator import ModelEvaluator
from Phase_5_ML.Predictor import Predictor
from Phase_5_ML.ModelVisualizer import ModelVisualizer


class MLPipeline:
    """
    MLPipeline Class

    The top-level orchestrator for the entire Machine Learning phase —
    matches the "MLPipeline" class at the top of the Complete Class
    Diagram. Coordinates ModelFactory, ModelTrainer, ModelEvaluator,
    Predictor and ModelVisualizer behind two distinct workflows:

        - User Mode      : a simplified, guided flow for non-technical
                            users. They upload data, pick a target
                            column, get a model recommendation, and the
                            pipeline handles training/evaluation/
                            prediction automatically.
        - Developer Mode  : full manual control over every step — model
                            choice, hyperparameters, feature engineering
                            options, preprocessing options, cross
                            validation strategy, etc.

    This mirrors the two flowcharts in the class diagram ("User Mode
    Flow" and "Developer Mode Flow").

    Attributes:
        mode (str)                : "user" or "developer".
        data (pd.DataFrame)       : The dataset currently loaded into the pipeline.
        target (str)              : Name of the target column.
        task_type (str)           : "classification", "regression" or "clustering",
                                     inferred automatically from the target column
                                     unless explicitly overridden.
        model_factory (ModelFactory)
        trainer (ModelTrainer)
        evaluator (ModelEvaluator)
        predictor (Predictor)
        visualizer (ModelVisualizer)
        results (dict)             : Cache of the most recent run's metrics/artifacts.

    Methods:
        run_pipeline(data, target, mode)     : Single entry point — dispatches
                                                  to run_user_mode() or
                                                  run_developer_mode() based on `mode`.
        run_user_mode(data, target)           : Simplified guided flow (see docstring).
        run_developer_mode(data, target, model_name, params, ...) :
                                                  Full manual-control flow.
        recommend_models(target)              : Thin wrapper around
                                                  ModelFactory.recommend_models()
                                                  using the pipeline's own data.
        train_pipeline(model_name, config)    : Build + train a model end-to-end.
        evaluate_pipeline()                    : Evaluate the currently trained model.
        predict_pipeline(input_data)            : Predict on new data using the
                                                  currently trained model.
        generate_report()                       : Produce a plain-dict summary of
                                                  the most recent run (model, metrics, mode).
    """

    def __init__(self, mode: str = "user"):
        if mode not in ("user", "developer"):
            raise ValueError("mode must be 'user' or 'developer'")

        self.mode = mode
        self.data = None
        self.target = None
        self.task_type = None

        self.model_factory = None
        self.trainer = None
        self.evaluator = None
        self.predictor = None
        self.visualizer = ModelVisualizer()

        self.results = {}

    # ── Task-type inference ──────────────────────────────────────────────
    @staticmethod
    def _infer_task_type(y: pd.Series) -> str:
        """
        Heuristically infer whether the target column implies a
        classification or regression problem.

        Rule of thumb: if the target is non-numeric, or numeric with a
        small number of unique values relative to the dataset size, it's
        almost certainly classification. Otherwise, regression.

        Args:
            y (pd.Series): The target column.

        Returns:
            str: "classification" or "regression".
        """
        if not pd.api.types.is_numeric_dtype(y):
            return "classification"

        n_unique = y.nunique()
        if n_unique <= max(20, int(0.05 * len(y))):
            return "classification"
        return "regression"

    # ── Single dispatcher entry point ───────────────────────────────────
    def run_pipeline(self, data: pd.DataFrame, target: str = None, **kwargs) -> dict:
        """
        Single entry point for running the pipeline end-to-end. Dispatches
        to run_user_mode() or run_developer_mode() based on self.mode.

        Args:
            data (pd.DataFrame): The dataset to train on.
            target (str, optional): Name of the target column. Required
                for classification/regression; omit for clustering.
            **kwargs: Forwarded to the underlying mode-specific method
                (e.g. model_name, params, test_size for developer mode).

        Returns:
            dict: The result of generate_report() after the run completes.
        """
        self.data = data
        self.target = target

        if self.mode == "user":
            self.run_user_mode(data, target)
        else:
            self.run_developer_mode(data, target, **kwargs)

        return self.generate_report()

    # ── User Mode: simplified, guided flow ───────────────────────────────
    def run_user_mode(self, data: pd.DataFrame, target: str = None, test_size: float = 0.2) -> dict:
        """
        Simplified flow for non-technical users, matching the "User Mode
        Flow" diagram:

            Upload Dataset -> Choose Target -> Get Recommended Models
            -> (auto-accept top recommendation) -> Train Model
            -> Evaluate Model -> Prediction-ready

        The pipeline picks the TOP recommended model automatically and
        trains it with sensible default hyperparameters — the user never
        has to touch a hyperparameter or understand what the model is.

        Args:
            data (pd.DataFrame): The dataset to train on.
            target (str, optional): Target column name. If omitted, the
                pipeline treats the task as clustering.
            test_size (float): Fraction of data held out for evaluation.

        Returns:
            dict: {"recommended_models": [...], "chosen_model": str,
                   "metrics": {...}}
        """
        self.data = data
        self.target = target
        self.task_type = (
            self._infer_task_type(data[target]) if target else "clustering"
        )

        self.model_factory = ModelFactory(task_type=self.task_type)
        recommendations = self.model_factory.recommend_models(self.task_type, data)
        top_choice = recommendations[0]["model_name"]

        if self.task_type == "clustering":
            model = self.model_factory.get_model(top_choice)
            model.fit(data)
            self.trainer = ModelTrainer(model=model, X_train=data)
            metrics = {"note": "Clustering models are evaluated via cluster labels, not supervised metrics."}
        else:
            X = data.drop(columns=[target])
            y = data[target]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            model = self.model_factory.get_model(top_choice)
            self.trainer = ModelTrainer(model=model, X_train=X_train, y_train=y_train)
            self.trainer.train()

            self.evaluator = ModelEvaluator(model=self.trainer.model, X_test=X_test, y_test=y_test)
            metrics = self.evaluator.get_all_metrics(task_type=self.task_type)
            self.predictor = Predictor(model=self.trainer.model)

        self.results = {
            "mode": "user",
            "recommended_models": recommendations,
            "chosen_model": top_choice,
            "task_type": self.task_type,
            "metrics": metrics,
        }
        return self.results

    # ── Developer Mode: full manual control ──────────────────────────────
    def run_developer_mode(
        self,
        data: pd.DataFrame,
        target: str = None,
        model_name: str = "random_forest",
        params: dict = None,
        test_size: float = 0.2,
        cv: int = 5,
        tuning: dict = None,
    ) -> dict:
        """
        Full manual-control flow for technical users, matching the
        "Developer Mode Flow" diagram:

            Upload Dataset -> Choose Target -> Choose Model
            -> Set Hyperparameters -> (Feature Engineering /
            Preprocessing handled upstream by Phase 1/2) -> Cross
            Validation -> Train Model -> Evaluate Model
            -> Prediction & Analysis

        Args:
            data (pd.DataFrame): The dataset to train on.
            target (str, optional): Target column name. Omit for clustering.
            model_name (str): Which model to build, e.g. "random_forest",
                "svm", "kmeans" — see ModelFactory.available_models().
            params (dict, optional): Hyperparameters for the chosen model.
            test_size (float): Fraction of data held out for evaluation
                (ignored for clustering).
            cv (int): Number of cross-validation folds to report alongside
                the final trained model's metrics.
            tuning (dict, optional): If provided, runs hyperparameter
                tuning before final training, e.g.
                {"method": "grid", "param_space": {...}}.

        Returns:
            dict: {"model_name": str, "params": dict, "cv_results": dict,
                   "tuning_results": dict | None, "metrics": dict}
        """
        params = params or {}
        self.data = data
        self.target = target
        self.task_type = (
            self._infer_task_type(data[target]) if target else "clustering"
        )

        self.model_factory = ModelFactory(task_type=self.task_type)
        model = self.model_factory.get_model(model_name, **params)

        if self.task_type == "clustering":
            model.fit(data)
            self.trainer = ModelTrainer(model=model, X_train=data)
            self.results = {
                "mode": "developer",
                "model_name": model_name,
                "params": params,
                "task_type": self.task_type,
                "metrics": {"note": "Clustering: inspect model.labels_ for cluster assignments."},
            }
            return self.results

        X = data.drop(columns=[target])
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        self.trainer = ModelTrainer(model=model, X_train=X_train, y_train=y_train)

        tuning_results = None
        if tuning:
            tuning_results = self.trainer.tune_hyperparameters(
                method=tuning.get("method", "grid"),
                param_space=tuning.get("param_space"),
                cv=cv,
            )
        else:
            self.trainer.train()

        cv_results = self.trainer.cross_validation(cv=cv)

        self.evaluator = ModelEvaluator(model=self.trainer.model, X_test=X_test, y_test=y_test)
        metrics = self.evaluator.get_all_metrics(task_type=self.task_type)
        self.predictor = Predictor(model=self.trainer.model)

        self.results = {
            "mode": "developer",
            "model_name": model_name,
            "params": params,
            "task_type": self.task_type,
            "cv_results": cv_results,
            "tuning_results": tuning_results,
            "metrics": metrics,
        }
        return self.results

    # ── Thin convenience wrappers used by both modes ─────────────────────
    def recommend_models(self, target: str) -> list:
        """
        Get model recommendations for the given target column, using the
        pipeline's currently loaded data.

        Args:
            target (str): Target column name.

        Returns:
            list[dict]: Same shape as ModelFactory.recommend_models().
        """
        if self.data is None:
            raise RuntimeError("No data loaded. Call run_pipeline()/run_user_mode() first.")

        task_type = self._infer_task_type(self.data[target])
        factory = ModelFactory(task_type=task_type)
        return factory.recommend_models(task_type, self.data)

    def train_pipeline(self, model_name: str, config: dict = None) -> object:
        """
        Build a model by name via ModelFactory and train it on the
        pipeline's current train split, in one call.

        Args:
            model_name (str): Model to build, e.g. "random_forest".
            config (dict, optional): Hyperparameters for the model.

        Returns:
            object: The trained model.
        """
        config = config or {}
        if self.model_factory is None:
            self.model_factory = ModelFactory(task_type=self.task_type or "classification")

        model = self.model_factory.get_model(model_name, **config)
        if self.trainer is None:
            raise RuntimeError("No training data set. Run run_user_mode()/run_developer_mode() first.")

        self.trainer.model = model
        return self.trainer.train()

    def evaluate_pipeline(self) -> dict:
        """
        Evaluate the currently trained model using the pipeline's stored
        evaluator.

        Returns:
            dict: Metrics dictionary (shape depends on task_type).
        """
        if self.evaluator is None:
            raise RuntimeError("No evaluator available. Run run_user_mode()/run_developer_mode() first.")
        return self.evaluator.get_all_metrics(task_type=self.task_type)

    def predict_pipeline(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict on new data using the currently trained model.

        Args:
            input_data (pd.DataFrame): New feature rows to predict on.

        Returns:
            pd.DataFrame: Output of Predictor.predict_batch().
        """
        if self.predictor is None:
            raise RuntimeError("No trained model available. Run run_user_mode()/run_developer_mode() first.")
        return self.predictor.predict_batch(input_data)

    # ── Reporting ─────────────────────────────────────────────────────────
    def generate_report(self) -> dict:
        """
        Produce a plain-dict summary of the most recent pipeline run —
        suitable for feeding into Phase 1's ReportGenerator or returning
        directly from a Flask API route as JSON.

        Returns:
            dict: self.results, the cached output of the last
            run_user_mode()/run_developer_mode() call.
        """
        if not self.results:
            raise RuntimeError("No pipeline run yet. Call run_pipeline() first.")
        return self.results
