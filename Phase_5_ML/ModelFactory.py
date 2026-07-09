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
        # `max_iter` is both a hardcoded default AND an exposed tunable
        # hyperparameter — merge instead of hardcoding as a literal kwarg,
        # otherwise setting max_iter from the Developer Mode form collides
        # with the hardcoded value and raises "multiple values for keyword
        # argument 'max_iter'".
        "logistic_regression": lambda **p: LogisticRegression(**{"max_iter": 1000, **p}),
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

    # ── Developer Mode hyperparameter definitions ───────────────────────
    # Each entry describes one control the frontend should render
    # (type/range/options), so the form only ever shows knobs that
    # actually apply to the chosen model instead of a fixed generic set.
    #
    # Expanded to expose the top 5+ highest-impact tuning knobs per model
    # (ordered most-impactful first), split per task type wherever a
    # classifier and regressor version of the same algorithm don't share
    # identical valid parameter values (e.g. `criterion`, `loss` options
    # differ between RandomForestClassifier and RandomForestRegressor).
    # Two models are intentionally left with fewer controls
    # (`linear_regression`, `naive_bayes`) because they genuinely don't
    # have more high-value tunable parameters in scikit-learn — adding
    # arbitrary ones there would just clutter the panel without giving
    # real tuning power.

    _TREE_ENSEMBLE_PARAMS_CLF = [
        {"name": "n_estimators", "label": "n_estimators", "type": "int", "default": 100, "min": 10, "max": 1000, "step": 10},
        {"name": "max_depth", "label": "max_depth", "type": "int", "default": None, "min": 1, "max": 100, "step": 1, "allow_empty": True},
        {"name": "min_samples_split", "label": "min_samples_split", "type": "int", "default": 2, "min": 2, "max": 50, "step": 1},
        {"name": "min_samples_leaf", "label": "min_samples_leaf", "type": "int", "default": 1, "min": 1, "max": 20, "step": 1},
        {"name": "max_features", "label": "max_features", "type": "select", "default": "sqrt", "options": ["sqrt", "log2", "None"]},
        {"name": "criterion", "label": "criterion", "type": "select", "default": "gini", "options": ["gini", "entropy", "log_loss"]},
    ]
    _TREE_ENSEMBLE_PARAMS_REG = [
        {"name": "n_estimators", "label": "n_estimators", "type": "int", "default": 100, "min": 10, "max": 1000, "step": 10},
        {"name": "max_depth", "label": "max_depth", "type": "int", "default": None, "min": 1, "max": 100, "step": 1, "allow_empty": True},
        {"name": "min_samples_split", "label": "min_samples_split", "type": "int", "default": 2, "min": 2, "max": 50, "step": 1},
        {"name": "min_samples_leaf", "label": "min_samples_leaf", "type": "int", "default": 1, "min": 1, "max": 20, "step": 1},
        {"name": "max_features", "label": "max_features", "type": "select", "default": "None", "options": ["sqrt", "log2", "None"]},
        {"name": "criterion", "label": "criterion", "type": "select", "default": "squared_error", "options": ["squared_error", "absolute_error", "friedman_mse", "poisson"]},
    ]

    _TREE_PARAMS_CLF = [
        {"name": "max_depth", "label": "max_depth", "type": "int", "default": None, "min": 1, "max": 100, "step": 1, "allow_empty": True},
        {"name": "min_samples_split", "label": "min_samples_split", "type": "int", "default": 2, "min": 2, "max": 50, "step": 1},
        {"name": "min_samples_leaf", "label": "min_samples_leaf", "type": "int", "default": 1, "min": 1, "max": 20, "step": 1},
        {"name": "criterion", "label": "criterion", "type": "select", "default": "gini", "options": ["gini", "entropy", "log_loss"]},
        {"name": "max_features", "label": "max_features", "type": "select", "default": "None", "options": ["sqrt", "log2", "None"]},
        {"name": "splitter", "label": "splitter", "type": "select", "default": "best", "options": ["best", "random"]},
    ]
    _TREE_PARAMS_REG = [
        {"name": "max_depth", "label": "max_depth", "type": "int", "default": None, "min": 1, "max": 100, "step": 1, "allow_empty": True},
        {"name": "min_samples_split", "label": "min_samples_split", "type": "int", "default": 2, "min": 2, "max": 50, "step": 1},
        {"name": "min_samples_leaf", "label": "min_samples_leaf", "type": "int", "default": 1, "min": 1, "max": 20, "step": 1},
        {"name": "criterion", "label": "criterion", "type": "select", "default": "squared_error", "options": ["squared_error", "friedman_mse", "absolute_error", "poisson"]},
        {"name": "max_features", "label": "max_features", "type": "select", "default": "None", "options": ["sqrt", "log2", "None"]},
        {"name": "splitter", "label": "splitter", "type": "select", "default": "best", "options": ["best", "random"]},
    ]

    _GRADIENT_BOOSTING_PARAMS_CLF = [
        {"name": "n_estimators", "label": "n_estimators", "type": "int", "default": 100, "min": 10, "max": 1000, "step": 10},
        {"name": "learning_rate", "label": "learning_rate", "type": "float", "default": 0.1, "min": 0.001, "max": 1.0, "step": 0.01},
        {"name": "max_depth", "label": "max_depth", "type": "int", "default": 3, "min": 1, "max": 20, "step": 1},
        {"name": "subsample", "label": "subsample", "type": "float", "default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05},
        {"name": "max_features", "label": "max_features", "type": "select", "default": "None", "options": ["sqrt", "log2", "None"]},
        {"name": "loss", "label": "loss", "type": "select", "default": "log_loss", "options": ["log_loss", "exponential"]},
    ]
    _GRADIENT_BOOSTING_PARAMS_REG = [
        {"name": "n_estimators", "label": "n_estimators", "type": "int", "default": 100, "min": 10, "max": 1000, "step": 10},
        {"name": "learning_rate", "label": "learning_rate", "type": "float", "default": 0.1, "min": 0.001, "max": 1.0, "step": 0.01},
        {"name": "max_depth", "label": "max_depth", "type": "int", "default": 3, "min": 1, "max": 20, "step": 1},
        {"name": "subsample", "label": "subsample", "type": "float", "default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05},
        {"name": "max_features", "label": "max_features", "type": "select", "default": "None", "options": ["sqrt", "log2", "None"]},
        {"name": "loss", "label": "loss", "type": "select", "default": "squared_error", "options": ["squared_error", "absolute_error", "huber", "quantile"]},
    ]

    _SVM_PARAMS_CLF = [
        {"name": "C", "label": "C (regularization)", "type": "float", "default": 1.0, "min": 0.001, "max": 100, "step": 0.1},
        {"name": "kernel", "label": "kernel", "type": "select", "default": "rbf", "options": ["linear", "rbf", "poly", "sigmoid"]},
        {"name": "gamma", "label": "gamma", "type": "select", "default": "scale", "options": ["scale", "auto"]},
        {"name": "degree", "label": "degree (poly kernel only)", "type": "int", "default": 3, "min": 1, "max": 10, "step": 1},
        {"name": "class_weight", "label": "class_weight", "type": "select", "default": "None", "options": ["None", "balanced"]},
    ]
    _SVM_PARAMS_REG = [
        {"name": "C", "label": "C (regularization)", "type": "float", "default": 1.0, "min": 0.001, "max": 100, "step": 0.1},
        {"name": "kernel", "label": "kernel", "type": "select", "default": "rbf", "options": ["linear", "rbf", "poly", "sigmoid"]},
        {"name": "gamma", "label": "gamma", "type": "select", "default": "scale", "options": ["scale", "auto"]},
        {"name": "degree", "label": "degree (poly kernel only)", "type": "int", "default": 3, "min": 1, "max": 10, "step": 1},
        {"name": "epsilon", "label": "epsilon", "type": "float", "default": 0.1, "min": 0.0, "max": 5.0, "step": 0.01},
    ]

    _KNN_PARAMS = [
        {"name": "n_neighbors", "label": "n_neighbors", "type": "int", "default": 5, "min": 1, "max": 50, "step": 1},
        {"name": "weights", "label": "weights", "type": "select", "default": "uniform", "options": ["uniform", "distance"]},
        {"name": "algorithm", "label": "algorithm", "type": "select", "default": "auto", "options": ["auto", "ball_tree", "kd_tree", "brute"]},
        {"name": "metric", "label": "metric", "type": "select", "default": "minkowski", "options": ["minkowski", "euclidean", "manhattan", "chebyshev"]},
        {"name": "p", "label": "p (power parameter for minkowski)", "type": "int", "default": 2, "min": 1, "max": 5, "step": 1},
    ]

    HYPERPARAM_SPECS = {
        "classification": {
            "logistic_regression": [
                {"name": "C", "label": "C (inverse regularization)", "type": "float", "default": 1.0, "min": 0.001, "max": 100, "step": 0.1},
                {"name": "max_iter", "label": "max_iter", "type": "int", "default": 1000, "min": 100, "max": 5000, "step": 100},
                {"name": "penalty", "label": "penalty", "type": "select", "default": "l2", "options": ["l2", "l1", "elasticnet", "none"]},
                {"name": "solver", "label": "solver", "type": "select", "default": "lbfgs", "options": ["lbfgs", "liblinear", "saga", "newton-cg", "sag"]},
                {"name": "class_weight", "label": "class_weight", "type": "select", "default": "None", "options": ["None", "balanced"]},
            ],
            "random_forest": _TREE_ENSEMBLE_PARAMS_CLF,
            "gradient_boosting": _GRADIENT_BOOSTING_PARAMS_CLF,
            "decision_tree": _TREE_PARAMS_CLF,
            "svm": _SVM_PARAMS_CLF,
            "knn": _KNN_PARAMS,
            "naive_bayes": [
                # GaussianNB genuinely only exposes one meaningful scalar
                # knob for tuning via a simple form (var_smoothing);
                # `priors` is an array input that doesn't map cleanly to
                # a single UI control, so it's intentionally left out.
                {"name": "var_smoothing", "label": "var_smoothing", "type": "float", "default": 1e-9, "min": 1e-12, "max": 1e-6, "step": 1e-9},
            ],
        },
        "regression": {
            "linear_regression": [
                # Ordinary least squares has no regularization/structural
                # knobs beyond these two — there isn't a meaningful 5th
                # parameter to add without switching to a different
                # estimator (e.g. Ridge/Lasso), which is out of scope here.
                {"name": "fit_intercept", "label": "fit_intercept", "type": "bool", "default": True},
                {"name": "positive", "label": "positive (force non-negative coefficients)", "type": "bool", "default": False},
            ],
            "random_forest": _TREE_ENSEMBLE_PARAMS_REG,
            "gradient_boosting": _GRADIENT_BOOSTING_PARAMS_REG,
            "decision_tree": _TREE_PARAMS_REG,
            "svm": _SVM_PARAMS_REG,
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
        # Filter out empty-string / None params (from form defaults), and
        # also a literal "None" STRING — several of the new select-type
        # hyperparameters (max_features, class_weight, ...) offer "None"
        # as a valid choice meaning "use scikit-learn's own default",
        # which is Python None, not the string "None". Omitting the key
        # entirely lets the estimator fall back to its real default
        # instead of erroring or silently passing the string through.
        clean_params = {}
        for k, v in params.items():
            if v in (None, ""):
                continue
            if isinstance(v, str) and v.strip().lower() == "none":
                continue
            clean_params[k] = v
        return registry[model_name](**clean_params)