import numpy as np
import pandas as pd


class Predictor:
    """
    Predictor Class

    The "production inference" layer — once a model has been trained and
    evaluated, this is what the rest of the system (and end users) use to
    actually get predictions out of it. Deliberately kept separate from
    ModelTrainer/ModelEvaluator so the prediction surface stays small,
    stable and safe to expose directly to a web API endpoint.

    Attributes:
        model (object): The trained model used for inference.

    Methods:
        predict(input_data)            : Predict on a full DataFrame.
        predict_proba(input_data)      : Predict class probabilities (classification only).
        predict_batch(input_data)      : Alias-friendly batch prediction with a tidy DataFrame output.
        predict_single(sample)         : Predict for one record, given as a dict.
        explain_prediction(sample)     : Lightweight explanation for a single prediction.
    """

    def __init__(self, model=None):
        self.model = model

    def _validate_model(self) -> None:
        if self.model is None:
            raise RuntimeError("No model assigned. Set self.model before predicting.")

    # ── Core prediction ───────────────────────────────────────────────────
    def predict(self, input_data: pd.DataFrame) -> pd.Series:
        """
        Predict the target value/label for every row in input_data.

        Args:
            input_data (pd.DataFrame): Feature rows, with the SAME columns
                (names, order, encoding) the model was trained on.

        Returns:
            pd.Series: Predicted values/labels, one per input row.
        """
        self._validate_model()
        predictions = self.model.predict(input_data)
        return pd.Series(predictions, index=input_data.index, name="prediction")

    # ── Probability prediction (classification only) ────────────────────
    def predict_proba(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict class probabilities for every row (classification models
        only). Useful for ranking predictions by confidence, setting a
        custom decision threshold, or computing ROC-AUC.

        Args:
            input_data (pd.DataFrame): Feature rows.

        Returns:
            pd.DataFrame: One column per class, containing the predicted
            probability of that class for each row.
        """
        self._validate_model()
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(
                "This model does not support predict_proba() "
                "(typical for regression models or some clustering models)."
            )

        proba = self.model.predict_proba(input_data)
        class_labels = getattr(self.model, "classes_", range(proba.shape[1]))
        return pd.DataFrame(
            proba, index=input_data.index, columns=[f"class_{c}" for c in class_labels]
        )

    # ── Batch prediction with a tidy combined output ────────────────────
    def predict_batch(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict on a batch of rows and return a tidy DataFrame combining
        the original input with the prediction column — convenient for
        downloading results or displaying them in a UI table.

        Args:
            input_data (pd.DataFrame): Feature rows.

        Returns:
            pd.DataFrame: input_data with an added "prediction" column
            (and, if available, a "confidence" column derived from
            predict_proba's max probability per row).
        """
        self._validate_model()
        result = input_data.copy()
        result["prediction"] = self.predict(input_data).values

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(input_data)
            result["confidence"] = proba.max(axis=1)

        return result

    # ── Single-record convenience prediction ─────────────────────────────
    def predict_single(self, sample: dict) -> dict:
        """
        Predict for a single record provided as a plain Python dict
        (e.g. straight from a JSON request body in a Flask route) —
        avoids the caller needing to manually wrap it in a DataFrame.

        Args:
            sample (dict): {feature_name: value, ...} for one record.

        Returns:
            dict: {
                "prediction": the predicted value/label,
                "probabilities": dict[class, float] (only if the model
                                  supports predict_proba)
            }
        """
        self._validate_model()
        df = pd.DataFrame([sample])

        result = {"prediction": self.model.predict(df)[0]}

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(df)[0]
            class_labels = getattr(self.model, "classes_", range(len(proba)))
            result["probabilities"] = {
                str(label): float(p) for label, p in zip(class_labels, proba)
            }

        return result

    # ── Lightweight per-prediction explanation ──────────────────────────
    def explain_prediction(self, sample: dict) -> dict:
        """
        Provide a lightweight explanation of why the model made a
        particular prediction for a single sample, using whatever
        introspection the model type supports:

            - Tree-based models (RandomForest, DecisionTree): ranks the
              sample's features by the model's global feature_importances_.
            - Linear models (LogisticRegression, LinearRegression): ranks
              features by |coefficient × feature_value|, approximating
              each feature's contribution to this specific prediction.
            - Any other model type: falls back to just returning the
              prediction with a note that detailed explanation isn't
              available for this model type.

        Note: this is intentionally simple and dependency-free. For
        more rigorous, model-agnostic explanations (e.g. SHAP values),
        that belongs in a dedicated Explainability phase — this method
        only provides a fast, "good enough" first look.

        Args:
            sample (dict): {feature_name: value, ...} for one record.

        Returns:
            dict: {
                "prediction": ...,
                "top_contributing_features": list[dict] | None,
                "note": str (present only when detailed explanation
                              isn't available for this model type)
            }
        """
        self._validate_model()
        df = pd.DataFrame([sample])
        prediction = self.model.predict(df)[0]

        # Tree-based ensembles / trees expose global feature importances.
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            ranked = sorted(
                zip(df.columns, importances), key=lambda x: x[1], reverse=True
            )
            return {
                "prediction": prediction,
                "top_contributing_features": [
                    {"feature": feat, "importance": float(imp)} for feat, imp in ranked
                ],
            }

        # Linear models expose coefficients per feature.
        if hasattr(self.model, "coef_"):
            coef = np.ravel(self.model.coef_)
            values = df.iloc[0].to_numpy(dtype=float)
            contributions = coef * values
            ranked = sorted(
                zip(df.columns, contributions), key=lambda x: abs(x[1]), reverse=True
            )
            return {
                "prediction": prediction,
                "top_contributing_features": [
                    {"feature": feat, "contribution": float(contrib)} for feat, contrib in ranked
                ],
            }

        return {
            "prediction": prediction,
            "top_contributing_features": None,
            "note": (
                "Detailed feature-level explanation is not available for "
                "this model type. Consider a model with "
                "feature_importances_ or coef_."
            ),
        }
