import os
import uuid
import pandas as pd

from Phase_5_ML.ModelTrainer import ModelTrainer
from Phase_5_ML.ModelEvaluator import ModelEvaluator
from Phase_5_ML.ModelFactory import ModelFactory
from Phase_5_ML.ModelVisualizer import ModelVisualizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)


class MLPipeline:
    """
    One entry point for both ML flows:

      run_auto()   — User Mode: guesses/accepts a target, auto-detects the
                     task type, quickly compares a small candidate pool of
                     fast models via cross-validation, and trains the best
                     one on the full train split. No knobs to turn.

      run_manual() — Developer Mode: explicit target, model, hyperparams,
                     vectorizer/scaling choices, test size, etc.

    Both return the same result shape so the frontend/report code and the
    RAG model_context stay identical regardless of which mode produced it.
    """

    def __init__(self, df: pd.DataFrame, plots_dir: str = None):
        self.df = df
        self.trainer = ModelTrainer(df)
        self.visualizer = ModelVisualizer(plots_dir=plots_dir)

    def _finish(self, task_type, model_name, model, scaler, X_test, y_test, feature_columns, target_column, extra=None):
        preds = model.predict(X_test)
        labels = sorted(pd.Series(list(y_test) + list(preds)).astype(str).unique()) if task_type == "classification" else None

        # PR-AUC needs predicted probabilities, not just hard labels. Every
        # classifier in ModelFactory exposes predict_proba (SVC is built
        # with probability=True specifically so this works), but this stays
        # defensive — if it's ever missing or errors for any reason, PR-AUC
        # is just omitted from metrics rather than breaking training.
        y_proba = None
        if task_type == "classification" and hasattr(model, "predict_proba"):
            try:
                proba_raw = model.predict_proba(X_test)
                model_classes = [str(c) for c in model.classes_]
                # Reindex probability columns into the same order as
                # `labels` — evaluate_classification assumes column i of
                # y_proba lines up with labels[i]. If a label shows up in
                # y_test/preds that the fitted model never actually saw as
                # a class, we skip PR-AUC entirely rather than guess.
                col_index = {c: i for i, c in enumerate(model_classes)}
                if all(lbl in col_index for lbl in labels):
                    y_proba = proba_raw[:, [col_index[lbl] for lbl in labels]]
            except Exception:
                y_proba = None

        metrics = ModelEvaluator.evaluate(
            task_type,
            [str(v) for v in y_test] if task_type == "classification" else y_test,
            [str(v) for v in preds] if task_type == "classification" else preds,
            labels=labels,
            y_proba=y_proba,
        )
        importance = ModelEvaluator.feature_importance(model, feature_columns)
        explanation = ModelEvaluator.explain(task_type, metrics, importance, target_column, model_name)

        plots = {}
        if task_type == "classification":
            plots["confusion_matrix"] = os.path.basename(
                self.visualizer.plot_confusion_matrix(metrics["confusion_matrix"], metrics["labels"])
            )
        else:
            plots["actual_vs_predicted"] = os.path.basename(
                self.visualizer.plot_actual_vs_predicted(list(y_test), list(preds))
            )
        if importance:
            plots["feature_importance"] = os.path.basename(
                self.visualizer.plot_feature_importance(importance)
            )

        model_id = uuid.uuid4().hex[:10]
        model_path = os.path.join(MODELS_DIR, f"model_{model_id}.pkl")
        ModelTrainer.save_model(
            model, scaler, model_path, feature_columns, task_type, model_name, target_column,
            labels=metrics.get("labels"),
        )

        result = {
            "model_id": model_id,
            "task_type": task_type,
            "model_name": model_name,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "metrics": metrics,
            "feature_importance": importance[:15],
            "explanation": explanation,
            "plots": plots,
            "train_size": len(self.df) - len(X_test),
            "test_size": len(X_test),
        }
        if extra:
            result.update(extra)
        return result

    # ── User Mode ──────────────────────────────────────────────────────
    def recommend_models(self, target_column: str) -> dict:
        """
        Preview step for User Mode: the PERSON has already chosen the target
        column (never auto-guessed here). This detects the task type,
        cross-validates the candidate pool, and returns a ranked list plus
        the top pick — WITHOUT training/saving a final model yet.
        """
        if not target_column:
            raise ValueError("Please choose a target column before requesting a recommendation.")
        task_type = self.trainer.detect_task_type(target_column)

        X, y, feature_columns = self.trainer.prepare_xy(target_column)
        if X.shape[1] == 0:
            raise ValueError("No usable numeric feature columns remain after excluding the target.")

        X_train, X_test, y_train, y_test = self.trainer.split(X, y, task_type=task_type)

        candidates = ModelFactory.AUTO_CANDIDATES[task_type]
        scores = []
        for name in candidates:
            mean_score, _ = self.trainer.cross_validate(task_type, name, X_train, y_train, cv=3)
            scores.append({"model": name, "score": mean_score if mean_score is not None else -1e9})

        ranked = sorted(scores, key=lambda r: r["score"], reverse=True)
        best = ranked[0]
        comparison_plot = os.path.basename(
            self.visualizer.plot_model_comparison(
                [s for s in scores if s["score"] != -1e9] or scores
            )
        )

        return {
            "target_column": target_column,
            "task_type": task_type,
            "feature_columns": feature_columns,
            "candidates": ranked,
            "recommended_model": best["model"],
            "plots": {"model_comparison": comparison_plot},
        }

    def run_auto(self, target_column: str, model_name: str = None) -> dict:
        """
        Trains the final User Mode model. `target_column` is REQUIRED — the
        person picks it, this never falls back to guessing it.

        If `model_name` is None, this re-runs the candidate comparison itself
        (still useful standalone) and picks the best one. If `model_name` is
        given (the person accepted the recommendation or picked an
        alternative from `recommend_models`), that exact model is trained
        directly — no redundant search.
        """
        if not target_column:
            raise ValueError("Please choose a target column before training.")
        task_type = self.trainer.detect_task_type(target_column)

        X, y, feature_columns = self.trainer.prepare_xy(target_column)
        if X.shape[1] == 0:
            raise ValueError("No usable numeric feature columns remain after excluding the target.")

        X_train, X_test, y_train, y_test = self.trainer.split(X, y, task_type=task_type)

        scores = None
        comparison_plot = None
        if model_name:
            chosen_model = model_name
            available = ModelFactory.available_models(task_type)
            if chosen_model not in available:
                raise ValueError(
                    f"'{chosen_model}' isn't a valid {task_type} model. "
                    f"Available: {available}"
                )
        else:
            candidates = ModelFactory.AUTO_CANDIDATES[task_type]
            scores = []
            for name in candidates:
                mean_score, _ = self.trainer.cross_validate(task_type, name, X_train, y_train, cv=3)
                scores.append({"model": name, "score": mean_score if mean_score is not None else -1e9})
            best = max(scores, key=lambda r: r["score"])
            chosen_model = best["model"]
            comparison_plot = os.path.basename(
                self.visualizer.plot_model_comparison(
                    [s for s in scores if s["score"] != -1e9] or scores
                )
            )

        model, scaler = self.trainer.train_one(task_type, chosen_model, X_train, y_train)

        result = self._finish(
            task_type, chosen_model, model, scaler, X_test, y_test, feature_columns, target_column,
            extra={
                "mode": "user",
                "target_guessed": False,
                "candidates_compared": scores or [],
                "user_selected_model": chosen_model,
            },
        )
        if comparison_plot:
            result["plots"]["model_comparison"] = comparison_plot
        return result

    # ── Developer Mode ────────────────────────────────────────────────
    def run_manual(
        self,
        target_column: str,
        model_name: str,
        task_type: str = None,
        feature_columns: list = None,
        test_size: float = 0.2,
        scale: bool = False,
        hyperparams: dict = None,
    ) -> dict:
        task_type = task_type or self.trainer.detect_task_type(target_column)
        X, y, used_columns = self.trainer.prepare_xy(target_column, feature_columns=feature_columns)
        if X.shape[1] == 0:
            raise ValueError("No usable numeric feature columns remain after excluding the target.")

        X_train, X_test, y_train, y_test = self.trainer.split(X, y, test_size=test_size, task_type=task_type)

        model, scaler = self.trainer.train_one(
            task_type, model_name, X_train, y_train, scale=scale, **(hyperparams or {})
        )

        if scaler is not None:
            X_test_for_eval = scaler.transform(X_test)
        else:
            X_test_for_eval = X_test

        return self._finish(
            task_type, model_name, model, scaler, X_test_for_eval, y_test, used_columns, target_column,
            extra={"mode": "developer", "target_guessed": False, "hyperparams": hyperparams or {}},
        )