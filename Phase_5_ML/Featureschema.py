from collections import Counter
import pandas as pd


class FeatureSchema:
    """
    Phase 5 trains on the ENCODED DataFrame (Phase 1's output), but the person
    picking a target/features or filling in a prediction form wants to see
    things the way they originally typed them — 'Male' / 'Female', not
    'Gender_Male': 1.

    This compares the raw (pre-cleaning, pre-encoding) DataFrame to the clean
    (post-cleaning, encoded) one and reconstructs the original identity of
    every encoded column, so the UI can show real category names/ranges and
    translate selections back to whatever the model actually expects.

    IMPORTANT — row alignment: `raw_df` is the dataset exactly as uploaded.
    `clean_df` has usually gone through de-duplication, null-handling and
    outlier removal, which can drop or reorder rows. Pairing them up
    position-by-position (a plain `zip`) silently produces WRONG category
    labels the moment row counts differ — which is virtually every real
    run. `_align` below pairs rows by their shared pandas index instead
    (which survives row-dropping, just not a `reset_index`), and falls back
    to positional pairing only when it's actually safe (identical length,
    no shared index at all). When neither is possible, the column is still
    always exposed — just flagged as `unreliable_mapping` instead of
    silently disappearing or showing corrupted categories.
    """

    @staticmethod
    def _align(raw_series: pd.Series, clean_series: pd.Series):
        """Best-effort, safe row alignment. Returns (raw_aligned, clean_aligned)
        or (None, None) if alignment can't be done safely."""
        common_idx = raw_series.index.intersection(clean_series.index)
        if len(common_idx) > 0:
            return raw_series.loc[common_idx], clean_series.loc[common_idx]
        if len(raw_series) == len(clean_series) and len(raw_series) > 0:
            return raw_series.reset_index(drop=True), clean_series.reset_index(drop=True)
        return None, None

    @staticmethod
    def _numeric_info(series: pd.Series) -> dict:
        numeric = pd.to_numeric(series, errors="coerce")
        return {
            "min": float(numeric.min()) if numeric.notna().any() else None,
            "max": float(numeric.max()) if numeric.notna().any() else None,
        }

    @staticmethod
    def build(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> dict:
        """Returns {clean_column_name: info} for EVERY column in clean_df —
        every clean column is guaranteed an entry; none are ever silently
        dropped from this schema."""
        schema = {}
        raw_cols = set(raw_df.columns) if raw_df is not None else set()

        # 1) Columns that exist in both frames under the same name.
        for col in clean_df.columns:
            if col not in raw_cols:
                continue
            raw_series = raw_df[col]
            clean_series = clean_df[col]

            if raw_series.dtype.kind in "iufcb":
                # Was already numeric — passed straight through, no encoding.
                info = FeatureSchema._numeric_info(clean_series)
                schema[col] = {
                    "type": "numeric",
                    "source_column": col,
                    "clean_columns": [col],
                    **info,
                }
                continue

            # Non-numeric in raw but ended up numeric in clean -> Label Encoding.
            if clean_series.dtype.kind in "iufc":
                raw_aligned, clean_aligned = FeatureSchema._align(raw_series, clean_series)

                if raw_aligned is None:
                    # Can't safely reconstruct which code means which label —
                    # still show the column (never drop it), just flagged.
                    codes = sorted(set(str(v) for v in clean_series.dropna().unique()))
                    schema[col] = {
                        "type": "categorical_label",
                        "source_column": col,
                        "clean_columns": [col],
                        "categories": codes,
                        "value_map": {c: c for c in codes},
                        "unreliable_mapping": True,
                    }
                    continue

                # Majority vote per code: robust even if a handful of rows
                # are still misaligned, without corrupting the whole mapping.
                votes = {}
                for raw_val, code in zip(raw_aligned, clean_aligned):
                    if pd.isna(code):
                        continue
                    key = str(int(code)) if float(code).is_integer() else str(code)
                    votes.setdefault(key, Counter())[str(raw_val)] += 1
                value_map = {k: counter.most_common(1)[0][0] for k, counter in votes.items()}

                schema[col] = {
                    "type": "categorical_label",
                    "source_column": col,
                    "clean_columns": [col],
                    "categories": sorted(set(value_map.values())),
                    "value_map": value_map,  # encoded code (str) -> original label
                }
            else:
                info = FeatureSchema._numeric_info(clean_series)
                schema[col] = {
                    "type": "numeric",
                    "source_column": col,
                    "clean_columns": [col],
                    **info,
                }

        # 2) One-Hot expanded columns: new in clean_df, named "<original>_<category>".
        #    Matched by the LONGEST raw column name that is a valid "prefix_"
        #    of the encoded column — not a naive single-underscore split —
        #    so this still works when the original column name itself
        #    contains underscores (e.g. "Marital_Status_Married").
        onehot_cols = [c for c in clean_df.columns if c not in raw_cols and c not in schema]
        groups = {}
        for col in onehot_cols:
            candidates = [rc for rc in raw_cols if col.startswith(f"{rc}_")]
            if candidates:
                prefix = max(candidates, key=len)
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

        # 3) Safety net — ANY clean_df column not covered above (no raw
        #    match, no one-hot prefix match) is still exposed, as plain
        #    numeric, so no column is ever silently missing from the schema.
        for col in clean_df.columns:
            if col not in schema:
                info = FeatureSchema._numeric_info(clean_df[col])
                schema[col] = {
                    "type": "numeric",
                    "source_column": col,
                    "clean_columns": [col],
                    **info,
                }

        return schema

    @staticmethod
    def missing_raw_columns(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> list:
        """
        Original (raw, as-uploaded) columns that have NO trace at all in
        clean_df — not passed through, not label-encoded, not one-hot
        expanded. Phase 5 never removes a column on its own; if a name
        shows up here it was dropped upstream in Phase 1 preprocessing
        (excluded manually, dropped as an ID column, or dropped as
        high-cardinality free text). Surfacing this list is what lets the
        person tell the difference between 'Phase 5 lost my feature' and
        'Phase 1 already removed it, here's which one and you can revisit
        that setting.'
        """
        if raw_df is None or clean_df is None:
            return []
        schema = FeatureSchema.build(raw_df, clean_df)
        surviving_sources = {info["source_column"] for info in schema.values()}
        return [c for c in raw_df.columns if c not in surviving_sources]

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
                    "unreliable_mapping": info.get("unreliable_mapping", False),
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