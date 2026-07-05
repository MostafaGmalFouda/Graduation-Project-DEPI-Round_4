import pandas as pd


class FeatureSchema:
    """
    Phase 5 trains on the ENCODED DataFrame (Phase 1's output), but the person
    picking features or filling in a prediction form wants to see things the
    way they originally typed them — 'Male' / 'Female', not 'Gender_Male':1.

    This compares the raw (pre-encoding) DataFrame to the clean (encoded) one
    and reconstructs that original identity for every encoded column, so the
    UI can show real category names/ranges and translate selections back to
    whatever the model actually expects.
    """

    @staticmethod
    def build(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> dict:
        """Returns {clean_column_name: info} for every column in clean_df."""
        schema = {}
        raw_cols = set(raw_df.columns) if raw_df is not None else set()

        # 1) Columns that exist in both frames under the same name.
        for col in clean_df.columns:
            if col in raw_cols:
                raw_series = raw_df[col]
                clean_series = clean_df[col]
                if raw_series.dtype.kind not in "iufcb" and clean_series.dtype.kind in "iufc":
                    # Same column, values turned into codes -> Label Encoding.
                    value_map = {}
                    for raw_val, code in zip(raw_series, clean_series):
                        if pd.isna(code):
                            continue
                        key = str(int(code)) if float(code).is_integer() else str(code)
                        value_map[key] = str(raw_val)
                    schema[col] = {
                        "type": "categorical_label",
                        "source_column": col,
                        "clean_columns": [col],
                        "categories": sorted(set(value_map.values())),
                        "value_map": value_map,  # encoded code (str) -> original label
                    }
                else:
                    numeric = pd.to_numeric(clean_series, errors="coerce")
                    schema[col] = {
                        "type": "numeric",
                        "source_column": col,
                        "clean_columns": [col],
                        "min": float(numeric.min()) if numeric.notna().any() else None,
                        "max": float(numeric.max()) if numeric.notna().any() else None,
                    }

        # 2) One-Hot expanded columns: new in clean_df, named "<original>_<category>".
        onehot_cols = [c for c in clean_df.columns if c not in raw_cols]
        groups = {}
        for col in onehot_cols:
            if "_" not in col:
                continue
            prefix = col.rsplit("_", 1)[0]
            if prefix in raw_cols:
                groups.setdefault(prefix, []).append(col)

        for source_column, cols in groups.items():
            for c in cols:
                category = c[len(source_column) + 1:]
                schema[c] = {
                    "type": "categorical_onehot",
                    "source_column": source_column,
                    "clean_columns": cols,
                    "this_category": category,
                }

        # 3) Anything left (no raw match, no one-hot prefix match) -> treat as
        #    a plain numeric feature; we simply can't reverse-map it.
        for col in onehot_cols:
            if col not in schema:
                numeric = pd.to_numeric(clean_df[col], errors="coerce")
                schema[col] = {
                    "type": "numeric",
                    "source_column": col,
                    "clean_columns": [col],
                    "min": float(numeric.min()) if numeric.notna().any() else None,
                    "max": float(numeric.max()) if numeric.notna().any() else None,
                }

        return schema

    @staticmethod
    def to_logical_features(schema: dict, clean_columns: list) -> list:
        """
        Collapses a list of encoded column names (e.g. a trained model's
        feature_columns) into the logical/original features a person should
        see — a One-Hot group becomes ONE entry with the original category
        options instead of N separate 0/1 columns.
        """
        seen_sources = set()
        logical = []
        for col in clean_columns:
            info = schema.get(col) or {
                "type": "numeric", "source_column": col, "clean_columns": [col],
                "min": None, "max": None,
            }
            source = info["source_column"]

            if info["type"] == "categorical_onehot":
                if source in seen_sources:
                    continue
                seen_sources.add(source)
                cols_here = [c for c in info["clean_columns"] if c in clean_columns]
                column_by_category = {schema[c]["this_category"]: c for c in cols_here}
                logical.append({
                    "name": source,
                    "type": "categorical",
                    "options": list(column_by_category.keys()),
                    "clean_columns": cols_here,
                    "column_by_category": column_by_category,
                })
            elif info["type"] == "categorical_label":
                logical.append({
                    "name": source,
                    "type": "categorical",
                    "options": info["categories"],
                    "clean_columns": [col],
                    "value_map": info["value_map"],
                })
            else:
                logical.append({
                    "name": source,
                    "type": "numeric",
                    "min": info.get("min"),
                    "max": info.get("max"),
                    "clean_columns": [col],
                })
        return logical