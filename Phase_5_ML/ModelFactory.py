from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB


class ModelFactory:
    """
    Central registry of every model BRight AI's ML phase can train, split
    by task type. Both the automatic (User) and manual (Developer) flows
    build models exclusively through this factory, so there is exactly one
    place that knows how to construct each algorithm.
    """

    CLASSIFICATION = {
        "logistic_regression": lambda **p: LogisticRegression(max_iter=1000, **p),
        "random_forest": lambda **p: RandomForestClassifier(random_state=42, **p),
        "gradient_boosting": lambda **p: GradientBoostingClassifier(random_state=42, **p),
        "decision_tree": lambda **p: DecisionTreeClassifier(random_state=42, **p),
        "svm": lambda **p: SVC(probability=True, **p),
        "knn": lambda **p: KNeighborsClassifier(**p),
        "naive_bayes": lambda **p: GaussianNB(**p),
    }

    REGRESSION = {
        "linear_regression": lambda **p: LinearRegression(**p),
        "random_forest": lambda **p: RandomForestRegressor(random_state=42, **p),
        "gradient_boosting": lambda **p: GradientBoostingRegressor(random_state=42, **p),
        "decision_tree": lambda **p: DecisionTreeRegressor(random_state=42, **p),
        "svm": lambda **p: SVR(**p),
        "knn": lambda **p: KNeighborsRegressor(**p),
    }

    # Sensible default candidate pool used by AutoML (User Mode) — fast
    # models only, so the "one click" experience stays quick.
    AUTO_CANDIDATES = {
        "classification": ["logistic_regression", "random_forest", "knn"],
        "regression": ["linear_regression", "random_forest", "knn"],
    }

    # Per-model hyperparameter definitions for Developer Mode. Each entry
    # describes one control the frontend should render (type/range/options),
    # so the form only ever shows knobs that actually apply to the chosen
    # model instead of a fixed generic set.
    _TREE_ENSEMBLE_PARAMS = [
        {"name": "n_estimators", "label": "n_estimators", "type": "int", "default": 100, "min": 10, "max": 1000, "step": 10},
        {"name": "max_depth", "label": "max_depth", "type": "int", "default": None, "min": 1, "max": 100, "step": 1, "allow_empty": True},
        {"name": "min_samples_split", "label": "min_samples_split", "type": "int", "default": 2, "min": 2, "max": 50, "step": 1},
        {"name": "min_samples_leaf", "label": "min_samples_leaf", "type": "int", "default": 1, "min": 1, "max": 20, "step": 1},
    ]
    _TREE_PARAMS = [
        {"name": "max_depth", "label": "max_depth", "type": "int", "default": None, "min": 1, "max": 100, "step": 1, "allow_empty": True},
        {"name": "min_samples_split", "label": "min_samples_split", "type": "int", "default": 2, "min": 2, "max": 50, "step": 1},
        {"name": "min_samples_leaf", "label": "min_samples_leaf", "type": "int", "default": 1, "min": 1, "max": 20, "step": 1},
    ]
    _GRADIENT_BOOSTING_PARAMS = [
        {"name": "n_estimators", "label": "n_estimators", "type": "int", "default": 100, "min": 10, "max": 1000, "step": 10},
        {"name": "learning_rate", "label": "learning_rate", "type": "float", "default": 0.1, "min": 0.001, "max": 1.0, "step": 0.01},
        {"name": "max_depth", "label": "max_depth", "type": "int", "default": 3, "min": 1, "max": 20, "step": 1},
        {"name": "subsample", "label": "subsample", "type": "float", "default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05},
    ]
    _SVM_PARAMS = [
        {"name": "C", "label": "C (regularization)", "type": "float", "default": 1.0, "min": 0.001, "max": 100, "step": 0.1},
        {"name": "kernel", "label": "kernel", "type": "select", "default": "rbf", "options": ["linear", "rbf", "poly", "sigmoid"]},
        {"name": "gamma", "label": "gamma", "type": "select", "default": "scale", "options": ["scale", "auto"]},
    ]
    _KNN_PARAMS = [
        {"name": "n_neighbors", "label": "n_neighbors", "type": "int", "default": 5, "min": 1, "max": 50, "step": 1},
        {"name": "weights", "label": "weights", "type": "select", "default": "uniform", "options": ["uniform", "distance"]},
        {"name": "algorithm", "label": "algorithm", "type": "select", "default": "auto", "options": ["auto", "ball_tree", "kd_tree", "brute"]},
    ]

    HYPERPARAM_SPECS = {
        "classification": {
            "logistic_regression": [
                {"name": "C", "label": "C (inverse regularization)", "type": "float", "default": 1.0, "min": 0.001, "max": 100, "step": 0.1},
                {"name": "max_iter", "label": "max_iter", "type": "int", "default": 1000, "min": 100, "max": 5000, "step": 100},
            ],
            "random_forest": _TREE_ENSEMBLE_PARAMS,
            "gradient_boosting": _GRADIENT_BOOSTING_PARAMS,
            "decision_tree": _TREE_PARAMS,
            "svm": _SVM_PARAMS,
            "knn": _KNN_PARAMS,
            "naive_bayes": [
                {"name": "var_smoothing", "label": "var_smoothing", "type": "float", "default": 1e-9, "min": 1e-12, "max": 1e-6, "step": 1e-9},
            ],
        },
        "regression": {
            "linear_regression": [
                {"name": "fit_intercept", "label": "fit_intercept", "type": "bool", "default": True},
                {"name": "positive", "label": "positive (force non-negative coefficients)", "type": "bool", "default": False},
            ],
            "random_forest": _TREE_ENSEMBLE_PARAMS,
            "gradient_boosting": _GRADIENT_BOOSTING_PARAMS,
            "decision_tree": _TREE_PARAMS,
            "svm": _SVM_PARAMS,
            "knn": _KNN_PARAMS,
        },
    }

    @classmethod
    def hyperparam_spec(cls, task_type: str, model_name: str) -> list:
        return cls.HYPERPARAM_SPECS.get(task_type, {}).get(model_name, [])

    @classmethod
    def available_models(cls, task_type: str) -> list:
        registry = cls.CLASSIFICATION if task_type == "classification" else cls.REGRESSION
        return list(registry.keys())

    @classmethod
    def build(cls, task_type: str, model_name: str, **params):
        registry = cls.CLASSIFICATION if task_type == "classification" else cls.REGRESSION
        if model_name not in registry:
            raise ValueError(f"Unknown model '{model_name}' for task '{task_type}'. "
                              f"Available: {list(registry.keys())}")
        # Filter out empty-string / None params (from form defaults)
        clean_params = {k: v for k, v in params.items() if v not in (None, "")}
        return registry[model_name](**clean_params)