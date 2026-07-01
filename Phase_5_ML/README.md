# 🔷 Phase 5 — Machine Learning Pipeline (User & Developer Modes)

Adds full machine learning capability on top of the EDA core — training,
evaluating, visualizing and serving predictions from classification,
regression and clustering models. Supports two distinct workflows:

- **User Mode** — guided, simplified flow for non-technical users.
- **Developer Mode** — full manual control over model choice,
  hyperparameters, and tuning strategy.

## 📦 Modules

| Module | Responsibility |
|---|---|
| `ModelFactory` | Recommends and creates models (RandomForest, Logistic/Linear Regression, SVM, KNN, DecisionTree, KMeans, DBSCAN) |
| `ModelTrainer` | Trains models, runs cross-validation, grid/random hyperparameter search, saves/loads models |
| `ModelEvaluator` | Computes classification metrics (accuracy, precision, recall, F1, ROC-AUC, confusion matrix) and regression metrics (MAE, MSE, RMSE, R²) |
| `Predictor` | Single/batch prediction, probability prediction, lightweight per-prediction explanation |
| `ModelVisualizer` | Confusion matrix, ROC/PR curves, feature importance, learning curves, actual vs predicted, distributions |
| `MLPipeline` | Top-level orchestrator tying everything together via `run_user_mode()` / `run_developer_mode()` |

## 🔄 Data Flow

```
[ Clean Data (Phase 1) ] → [ ModelFactory ] → model
   → [ ModelTrainer ] → trained model
   → [ ModelEvaluator ] → metrics
   → [ Predictor ] → predictions
   → [ ModelVisualizer ] → plots
```

## 🚀 Quick Start

### User Mode (simplified)
```python
import pandas as pd
from Phase_5_ML.MLPipeline import MLPipeline

df = pd.read_csv("train.csv")
pipeline = MLPipeline(mode="user")
result = pipeline.run_pipeline(df, target="label")

print(result["chosen_model"])   # the auto-selected best model
print(result["metrics"])        # full metrics dict
```

### Developer Mode (full control)
```python
pipeline = MLPipeline(mode="developer")
result = pipeline.run_developer_mode(
    df,
    target="label",
    model_name="random_forest",
    params={"n_estimators": 200, "max_depth": 10},
    tuning={"method": "grid", "param_space": {"n_estimators": [100, 200], "max_depth": [5, 10, None]}},
    cv=5,
)
print(result["tuning_results"]["best_params"])
print(result["metrics"])
```

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

## 🧩 Integration Notes

- `MLPipeline` automatically infers `task_type` ("classification" /
  "regression" / "clustering") from the target column — pass `target=None`
  for clustering.
- `ModelFactory.recommend_models()` is what powers User Mode's "Get
  Recommended Models" step — it returns a ranked list with
  plain-language reasoning, not just model names.
- `ModelVisualizer` uses a non-interactive matplotlib backend ("Agg"),
  so it's safe to call from a Flask route — use `save_plot()` to persist
  the figure and serve it as a static file rather than `show_plots()`.
- `Predictor.explain_prediction()` is a lightweight, dependency-free
  explanation (feature importances / coefficients). For more rigorous,
  model-agnostic explanations (SHAP/LIME), that belongs in a future
  dedicated Explainability phase.
