import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)


class ModelTrainer:
    """
    ModelTrainer Class

    Responsible for everything that happens to a model BETWEEN "it was
    just created by ModelFactory" and "it is ready to be evaluated by
    ModelEvaluator": fitting it on data, validating it, tuning its
    hyperparameters, and persisting it to disk.

    Design notes:
        - The trainer is intentionally model-agnostic: it works with any
          scikit-learn-compatible estimator that implements .fit() —
          whether it came from ModelFactory or was constructed manually.
        - tune_hyperparameters() is a convenience wrapper that picks
          grid_search() or random_search() based on a simple method
          flag, so MLPipeline (the orchestrator) can call a single
          method regardless of which tuning strategy "Developer Mode"
          configured.

    Attributes:
        model (object)       : The estimator being trained (set in
                                set_parameters() / train(), or passed
                                directly to __init__).
        X_train (DataFrame)  : Training features.
        y_train (Series)     : Training target.
        config (dict)        : Arbitrary training configuration, e.g.
                                {"cv": 5, "scoring": "accuracy"}.

    Methods:
        train()                                 : Fit the model on X_train/y_train.
        cross_validation(cv)                    : K-fold cross-validation score.
        grid_search(param_grid)                 : Exhaustive hyperparameter search.
        random_search(param_dist)               : Randomized hyperparameter search.
        tune_hyperparameters(method)            : Unified entry point for tuning.
        set_parameters(params)                  : Update the model's hyperparameters in place.
        save_model(path)                         : Persist the trained model to disk.
        load_model(path)                         : Load a previously saved model.
    """

    def __init__(self, model=None, X_train: pd.DataFrame = None, y_train: pd.Series = None, config: dict = None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.config = config or {}
        self._search_result = None  # cached GridSearchCV / RandomizedSearchCV object

    # ── Basic training ───────────────────────────────────────────────────
    def train(self):
        """
        Fit the model on the stored training data.

        Returns:
            object: The fitted model (same instance as self.model).
        """
        self._validate_ready_to_train()
        self.model.fit(self.X_train, self.y_train)
        return self.model

    # ── Cross-validation ──────────────────────────────────────────────────
    def cross_validation(self, cv: int = 5, scoring: str = None) -> dict:
        """
        Evaluate the model's generalization performance using k-fold
        cross-validation, WITHOUT permanently fitting it on the full
        training set.

        Why this matters: a single train/test split can be misleading
        on small or unevenly distributed datasets — cross-validation
        gives a more reliable estimate of how the model performs on
        unseen data by training/testing on several different splits.

        Args:
            cv (int): Number of folds.
            scoring (str, optional): scikit-learn scoring string (e.g.
                "accuracy", "f1", "r2"). If None, uses the model's
                default scorer.

        Returns:
            dict: {
                "scores": list[float]  -> score for each fold,
                "mean": float,
                "std": float
            }
        """
        self._validate_ready_to_train()

        scores = cross_val_score(
            self.model, self.X_train, self.y_train, cv=cv, scoring=scoring
        )
        return {
            "scores": scores.tolist(),
            "mean": float(scores.mean()),
            "std": float(scores.std()),
        }

    # ── Exhaustive grid search ───────────────────────────────────────────
    def grid_search(self, param_grid: dict, cv: int = 5, scoring: str = None) -> dict:
        """
        Exhaustively try every combination of hyperparameters in
        `param_grid` and keep the best-performing one, validated with
        cross-validation.

        Best when: the hyperparameter space is small enough to fully
        enumerate (few parameters, few values each) and you want a
        guarantee of finding the best combination within that space.

        Args:
            param_grid (dict): e.g. {"n_estimators": [50, 100, 200],
                                       "max_depth": [None, 5, 10]}
            cv (int): Number of cross-validation folds.
            scoring (str, optional): scikit-learn scoring string.

        Returns:
            dict: {
                "best_params": dict,
                "best_score": float,
                "best_estimator": object  -> the best fitted model
            }
        """
        self._validate_ready_to_train()

        search = GridSearchCV(
            self.model, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1
        )
        search.fit(self.X_train, self.y_train)

        self._search_result = search
        self.model = search.best_estimator_

        return {
            "best_params": search.best_params_,
            "best_score": float(search.best_score_),
            "best_estimator": search.best_estimator_,
        }

    # ── Randomized search ────────────────────────────────────────────────
    def random_search(self, param_dist: dict, n_iter: int = 20, cv: int = 5, scoring: str = None) -> dict:
        """
        Sample `n_iter` random combinations from `param_dist` and keep
        the best-performing one, validated with cross-validation.

        Best when: the hyperparameter space is too large to fully
        enumerate (e.g. continuous ranges, many parameters) — random
        search explores the space efficiently with a fixed compute budget.

        Args:
            param_dist (dict): e.g. {"n_estimators": range(50, 500),
                                       "max_depth": [None, 5, 10, 20]}
            n_iter (int): Number of random parameter combinations to try.
            cv (int): Number of cross-validation folds.
            scoring (str, optional): scikit-learn scoring string.

        Returns:
            dict: {
                "best_params": dict,
                "best_score": float,
                "best_estimator": object
            }
        """
        self._validate_ready_to_train()

        search = RandomizedSearchCV(
            self.model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=42,
        )
        search.fit(self.X_train, self.y_train)

        self._search_result = search
        self.model = search.best_estimator_

        return {
            "best_params": search.best_params_,
            "best_score": float(search.best_score_),
            "best_estimator": search.best_estimator_,
        }

    # ── Unified tuning entry point ───────────────────────────────────────
    def tune_hyperparameters(self, method: str = "grid", param_space: dict = None, **kwargs) -> dict:
        """
        Single entry point for hyperparameter tuning — picks the right
        underlying strategy based on `method`. Exists so callers (like
        MLPipeline in Developer Mode) don't need to know which specific
        search method is being used under the hood.

        Args:
            method (str): "grid" or "random".
            param_space (dict): The parameter grid/distribution to search.
            **kwargs: Forwarded to grid_search() / random_search()
                (e.g. cv, scoring, n_iter).

        Returns:
            dict: Same shape as grid_search()/random_search() output.
        """
        if param_space is None:
            raise ValueError("param_space is required for hyperparameter tuning")

        if method == "grid":
            return self.grid_search(param_space, **kwargs)
        elif method == "random":
            return self.random_search(param_space, **kwargs)
        else:
            raise ValueError("method must be either 'grid' or 'random'")

    # ── Manual hyperparameter updates ───────────────────────────────────
    def set_parameters(self, params: dict) -> None:
        """
        Update the current model's hyperparameters in place, without
        creating a new model instance. Useful for "Developer Mode" UIs
        where a user adjusts sliders/inputs for an already-selected
        model.

        Args:
            params (dict): Hyperparameters to set, forwarded to the
                model's set_params() method.

        Returns:
            None
        """
        if self.model is None:
            raise RuntimeError("No model is set. Assign self.model before calling set_parameters().")
        self.model.set_params(**params)

    # ── Persistence ───────────────────────────────────────────────────────
    def save_model(self, path: str) -> None:
        """
        Persist the current (trained) model to disk using joblib, which
        handles scikit-learn estimators (including their internal numpy
        arrays) more efficiently than plain pickle.

        Args:
            path (str): Destination file path (e.g. "models/rf_v1.joblib").

        Returns:
            None
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train or assign a model first.")
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        """
        Load a previously saved model from disk and set it as the
        trainer's active model.

        Args:
            path (str): Path to a .joblib (or .pkl) file saved via
                save_model().

        Returns:
            object: The loaded model.
        """
        self.model = joblib.load(path)
        return self.model

    # ── Internal validation helper ───────────────────────────────────────
    def _validate_ready_to_train(self) -> None:
        if self.model is None:
            raise RuntimeError("No model assigned. Set self.model before training.")
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("X_train and y_train must both be set before training.")
