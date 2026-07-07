import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)


class ModelEvaluator:
    """Computes and formats metrics for a fitted model's test-set predictions."""

    @staticmethod
    def evaluate_classification(y_true, y_pred, labels=None) -> dict:
            labels = labels or sorted(set(list(y_true) + list(y_pred)))
            acc = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="weighted", zero_division=0
            )
            cm = confusion_matrix(y_true, y_pred, labels=labels)

            # ── Per-class breakdown + class-balance measurements ────────────────
            # A single "weighted" score can look perfectly fine even when a model
            # is quietly just predicting the majority class every time — that
            # only shows up once you look at each class individually and at how
            # skewed the class counts are to begin with.
            p_per, r_per, f1_per, support_per = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0, labels=labels
            )
            per_class = [
                {
                    "label": str(lbl),
                    "precision": round(float(p), 4),
                    "recall": round(float(r), 4),
                    "f1_score": round(float(f), 4),
                    "support": int(s),
                }
                for lbl, p, r, f, s in zip(labels, p_per, r_per, f1_per, support_per)
            ]

            class_counts = {str(lbl): int(support_per[i]) for i, lbl in enumerate(labels)}
            max_count = max(support_per) if len(support_per) else 0
            min_count = min(support_per) if len(support_per) else 0
            imbalance_ratio = round(float(min_count / max_count), 4) if max_count > 0 else 1.0
            # Rule of thumb: a minority class under ~40% the size of the largest
            # class is worth flagging — weighted accuracy alone won't show it.
            is_imbalanced = imbalance_ratio < 0.4

            return {
                "task_type": "classification",
                "accuracy": round(float(acc), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1_score": round(float(f1), 4),
                "confusion_matrix": cm.tolist(),
                "labels": [str(l) for l in labels],
                "per_class": per_class,
                "class_distribution": class_counts,
                "imbalance_ratio": imbalance_ratio,
                "is_imbalanced": is_imbalanced,
            }

    @staticmethod
    def evaluate_regression(y_true, y_pred) -> dict:
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_true, y_pred)
        return {
            "task_type": "regression",
            "mae": round(float(mae), 4),
            "mse": round(float(mse), 4),
            "rmse": round(rmse, 4),
            "r2_score": round(float(r2), 4),
        }

    @staticmethod
    def evaluate(task_type: str, y_true, y_pred, labels=None) -> dict:
        if task_type == "classification":
            return ModelEvaluator.evaluate_classification(y_true, y_pred, labels=labels)
        return ModelEvaluator.evaluate_regression(y_true, y_pred)

    @staticmethod
    def feature_importance(model, feature_columns: list) -> list:
        """Returns a sorted [{feature, importance}] list, or [] if the model
        doesn't expose one (e.g. KNN, SVM without a linear kernel)."""
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)

        if importances is None:
            return []

        ranked = sorted(zip(feature_columns, importances), key=lambda x: x[1], reverse=True)
        return [{"feature": f, "importance": round(float(i), 4)} for f, i in ranked]

    @staticmethod
    def explain(task_type: str, metrics: dict, importance: list, target_column: str, model_name: str) -> str:
        """
        Rule-based, deterministic explanation of what the numbers actually
        mean — not a black box call to an external model, so it never
        hallucinates a number that isn't in `metrics`. Every sentence is
        generated directly from a specific field already computed above.
        """
        model_label = model_name.replace("_", " ").title()
        parts = []

        if task_type == "classification":
            acc = metrics["accuracy"]
            bal_acc = metrics["balanced_accuracy"]
            parts.append(
                f"The {model_label} model predicts '{target_column}' correctly {acc:.0%} of the time "
                f"on data it never saw during training."
            )

            if metrics.get("is_imbalanced"):
                dist = metrics["class_distribution"]
                biggest = max(dist, key=dist.get)
                smallest = min(dist, key=dist.get)
                gap_note = (
                    f"This dataset is imbalanced — class '{biggest}' has {dist[biggest]} examples in the "
                    f"test set versus only {dist[smallest]} for '{smallest}'. Because of that, plain accuracy "
                    f"can look better than the model really is (a model that just always guessed '{biggest}' "
                    f"could still score well). The balanced accuracy of {bal_acc:.0%} — which weighs every "
                    f"class equally regardless of size — is the more honest number here."
                )
                parts.append(gap_note)

                worst = min(metrics["per_class"], key=lambda c: c["f1_score"])
                if worst["f1_score"] < 0.6:
                    parts.append(
                        f"The model struggles most with class '{worst['label']}' "
                        f"(F1-score {worst['f1_score']:.0%}, only {worst['support']} examples to learn from) — "
                        f"more labeled data for that class would likely help more than tuning hyperparameters."
                    )
            else:
                parts.append(
                    f"Class sizes in the test set are reasonably balanced, so accuracy ({acc:.0%}) and "
                    f"balanced accuracy ({bal_acc:.0%}) are close together and both trustworthy."
                )
        else:
            r2 = metrics["r2_score"]
            rmse = metrics["rmse"]
            fit_quality = (
                "an excellent" if r2 >= 0.9 else
                "a good" if r2 >= 0.7 else
                "a moderate" if r2 >= 0.4 else
                "a weak"
            )
            parts.append(
                f"The {model_label} model explains {fit_quality} share of the variation in "
                f"'{target_column}' (R² = {r2:.2f}). On average, its predictions are off by about "
                f"{rmse:.2f} units (RMSE)."
            )
            if r2 < 0.4:
                parts.append(
                    "That low R² suggests the current features may not capture what actually drives "
                    f"'{target_column}' — consider adding more relevant columns before tuning the model further."
                )

        if importance:
            top = importance[:3]
            names = ", ".join(f"'{f['feature']}'" for f in top)
            parts.append(f"The features driving these predictions the most are {names}.")
        else:
            parts.append(
                f"{model_label} doesn't expose feature importances directly, so no ranking of "
                "influential columns is available for this particular model."
            )

        return " ".join(parts)
