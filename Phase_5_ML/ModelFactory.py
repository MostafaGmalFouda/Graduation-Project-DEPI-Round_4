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
