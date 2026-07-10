import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class DataPreprocessor:
    """
    Smart preprocessing pipeline for tabular data.

    Steps (recommended order):
        1. exclude_columns        — drop user-specified columns
        2. drop_id_columns        — drop numeric ID-like columns (optional)
        3. clean_empty_strings    — normalize whitespace-only text to NaN
        4. remove_empty_rows      — drop fully-empty rows
        5. remove_constant_columns— drop zero-information columns
        6. handle_nulls           — drop/fill remaining missing values
        7. handle_text_columns    — drop/hash ID-like text columns
        8. convert_types          — trim, datetime detection, numeric downcast
        9. remove_duplicates      — exact-row deduplication
       10. auto_encode            — smart encoding for categoricals (USER MODE)
       11. encode_categoricals    — manual encoding override (DEVELOPER MODE)
       12. handle_outliers        — handle outliers (IQR/Z‑score, cap/remove)
       13. get_clean_data         — return the processed DataFrame
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

    # ──────────────────────────────────────────────────────────────
    # 1. Exclude Columns
    # ──────────────────────────────────────────────────────────────
    def exclude_columns(self, columns: list) -> pd.DataFrame:
        """Drop user-specified columns. Silently ignores missing names."""
        if not columns:
            return self.data
        existing = [c for c in columns if c in self.data.columns]
        if existing:
            self.data.drop(columns=existing, inplace=True)
            print(f"[exclude_columns] Dropped {len(existing)} column(s): {existing}")
        return self.data

    # ──────────────────────────────────────────────────────────────
    # 2. Drop Numeric ID Columns (new from v2, now active)
    # ──────────────────────────────────────────────────────────────
    def drop_id_columns(self) -> pd.DataFrame:
        """
        Detect and drop numeric columns that are clearly row identifiers.
        A column is considered an ID if:
            - All values are unique (n_unique == n_rows)
            AND
            - Values are sequential integers (max - min + 1 == n_rows)
            OR
            - Column name contains a known ID keyword.
        """
        id_keywords = {
            "id", "index", "rownum", "row_num", "passengerid",
            "customerid", "userid", "number", "num", "no", "code"
        }

        numeric_cols = self.data.select_dtypes(include=["number"]).columns.tolist()
        dropped = []

        for col in numeric_cols:
            n_rows = len(self.data)
            n_unique = self.data[col].nunique()
            if n_unique != n_rows:          # must be fully unique
                continue

            col_lower = col.lower().replace("_", "").replace(" ", "")
            has_keyword = any(kw in col_lower for kw in id_keywords)

            try:
                is_sequential = (self.data[col].max() - self.data[col].min() + 1) == n_rows
            except Exception:
                is_sequential = False

            if has_keyword or is_sequential:
                self.data.drop(columns=[col], inplace=True)
                dropped.append(col)
                print(f"[drop_id_columns] Dropped '{col}' "
                      f"({n_unique}/{n_rows} unique, sequential={is_sequential}).")

        if not dropped:
            print("[drop_id_columns] No numeric ID columns detected.")
        return self.data

    # ──────────────────────────────────────────────────────────────
    # 3. Clean Empty Strings (from v1)
    # ──────────────────────────────────────────────────────────────
    def clean_empty_strings(self) -> pd.DataFrame:
        """
        Normalize whitespace-only / empty-string cells in text columns to
        real NaN values, so subsequent steps can detect them.
        """
        obj_cols = self.data.select_dtypes(include=["object", "string"]).columns
        if len(obj_cols) == 0:
            return self.data

        before_non_null = self.data[obj_cols].notna().sum().sum()
        for col in obj_cols:
            self.data[col] = self.data[col].astype(str).replace(
                r"^\s*$", pd.NA, regex=True
            )
            # astype(str) turns real NaNs into the string "nan" — undo that
            self.data[col] = self.data[col].replace("nan", pd.NA)
        after_non_null = self.data[obj_cols].notna().sum().sum()

        converted = before_non_null - after_non_null
        if converted > 0:
            print(f"[clean_empty_strings] Converted {int(converted)} empty/whitespace "
                  f"cell(s) to NaN across {len(obj_cols)} text column(s).")
        return self.data

    # ──────────────────────────────────────────────────────────────
    # 4. Remove Empty Rows (from v1)
    # ──────────────────────────────────────────────────────────────
    def remove_empty_rows(self) -> pd.DataFrame:
        """Drop rows that are entirely empty (NaN across every column)."""
        before = len(self.data)
        self.data.dropna(how="all", inplace=True)
        removed = before - len(self.data)
        if removed:
            print(f"[remove_empty_rows] Removed {removed} fully empty row(s).")
        else:
            print("[remove_empty_rows] No fully empty rows found.")
        return self.data

    # ──────────────────────────────────────────────────────────────
    # 5. Remove Constant Columns (from v1)
    # ──────────────────────────────────────────────────────────────
    def remove_constant_columns(self) -> pd.DataFrame:
        """
        Drop columns with zero informational value: only one unique
        non-null value (or completely empty).
        """
        constant_cols = [
            c for c in self.data.columns
            if self.data[c].nunique(dropna=True) <= 1
        ]
        if constant_cols:
            self.data.drop(columns=constant_cols, inplace=True)
            print(f"[remove_constant_columns] Dropped constant column(s): {constant_cols}")
        else:
            print("[remove_constant_columns] No constant columns found.")
        return self.data

    # ──────────────────────────────────────────────────────────────
    # 6. Handle Nulls
    # ──────────────────────────────────────────────────────────────
    def handle_nulls(self, threshold: float = 0.4, fill_strategy: str = "median") -> pd.DataFrame:
        """
        - null ratio > threshold  → drop column
        - null ratio < 5%         → drop rows
        - in between              → fill (median/mean/mode for numeric, mode for categorical)
        """
        if fill_strategy not in ("median", "mean", "mode"):
            fill_strategy = "median"

        for col in list(self.data.columns):
            null_count = self.data[col].isnull().sum()
            if null_count == 0:
                continue

            null_ratio = null_count / len(self.data)

            if null_ratio > threshold:
                self.data.drop(columns=[col], inplace=True)
                print(f"[handle_nulls] Dropped column '{col}' — {null_ratio:.0%} nulls.")

            elif null_ratio < 0.05:
                self.data.dropna(subset=[col], inplace=True)
                print(f"[handle_nulls] Dropped rows with nulls in '{col}' ({null_ratio:.0%}).")

            else:
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    if fill_strategy == "mean":
                        fill_val = self.data[col].mean()
                    elif fill_strategy == "mode":
                        fill_val = self.data[col].mode()[0]
                    else:
                        fill_val = self.data[col].median()
                    self.data[col] = self.data[col].fillna(fill_val)
                    print(f"[handle_nulls] Filled '{col}' with {fill_strategy}: {fill_val:.4g}.")
                else:
                    mode_series = self.data[col].mode()
                    if mode_series.empty:
                        continue
                    mode_val = mode_series[0]
                    self.data[col] = self.data[col].fillna(mode_val)
                    print(f"[handle_nulls] Filled '{col}' with mode: '{mode_val}'.")

        return self.data

    # ──────────────────────────────────────────────────────────────
    # 7. Handle Text Columns (ID-like / free-text)
    # ──────────────────────────────────────────────────────────────
    def handle_text_columns(
        self,
        unique_threshold: float = 0.6,
        action: str = "drop"
    ) -> pd.DataFrame:
        """
        Detect and handle high-cardinality text columns useless for modelling
        (e.g. Name, Ticket, Cabin).

        Args:
            unique_threshold : n_unique/n_rows >= this → ID-like. Default 0.6.
            action           : 'drop' | 'hash' | 'keep'
        """
        if action not in ("drop", "hash", "keep"):
            action = "drop"

        text_cols = self.data.select_dtypes(include=["object", "string"]).columns.tolist()

        if not text_cols:
            print("[handle_text_columns] No text columns found.")
            return self.data

        for col in list(text_cols):
            if col not in self.data.columns:
                continue

            n_rows = len(self.data)
            n_unique = self.data[col].nunique(dropna=True)
            if n_rows == 0:
                continue

            ratio = n_unique / n_rows
            is_mixed_alphanum = self._is_mixed_alphanum(col)

            if ratio >= unique_threshold or is_mixed_alphanum:
                if action == "drop":
                    self.data.drop(columns=[col], inplace=True)
                    reason = "mixed alphanumeric" if is_mixed_alphanum else f"{ratio:.0%} unique"
                    print(f"[handle_text_columns] Dropped '{col}' — ID-like ({reason}).")
                elif action == "hash":
                    self.data[col] = self.data[col].apply(lambda x: hash(str(x)) % 10_000)
                    print(f"[handle_text_columns] Hashed '{col}' → numeric.")
                else:
                    print(f"[handle_text_columns] Keeping '{col}' as-is.")
            else:
                print(
                    f"[handle_text_columns] '{col}': {n_unique} unique ({ratio:.0%}) "
                    f"— low-cardinality, will be encoded."
                )

        return self.data

    def _is_mixed_alphanum(self, col: str) -> bool:
        """Returns True if column values mostly contain both letters and digits (ticket/ID-like)."""
        try:
            sample = self.data[col].dropna().astype(str).head(50)
            if len(sample) == 0:
                return False
            mixed = sample.apply(
                lambda x: any(c.isalpha() for c in x) and any(c.isdigit() for c in x)
            ).sum()
            return (mixed / len(sample)) >= 0.3
        except Exception:
            return False

    # ──────────────────────────────────────────────────────────────
    # 8. Convert Types
    # ──────────────────────────────────────────────────────────────
    def convert_types(self, schema: dict = None) -> pd.DataFrame:
        """
        - Trim whitespace from object columns.
        - Schema-based conversion if provided.
        - Auto-detect datetime strings (≥80% parseable).
        - Downcast numerics to save memory.
        """
        # Trim whitespace
        for col in self.data.select_dtypes(include=["object"]).columns:
            self.data[col] = self.data[col].astype(str).str.strip()

        # Schema-based conversion
        if schema:
            for col, dtype in schema.items():
                if col in self.data.columns:
                    try:
                        self.data[col] = self.data[col].astype(dtype)
                    except (ValueError, TypeError) as e:
                        print(f"[convert_types] Cannot convert '{col}' to {dtype}: {e}")

        # Auto datetime + numeric downcast
        for col in self.data.columns:
            if self.data[col].dtype == "object":
                try:
                    converted = pd.to_datetime(self.data[col], infer_datetime_format=True)
                    if converted.notna().mean() >= 0.8:
                        self.data[col] = converted
                        print(f"[convert_types] Converted '{col}' → datetime.")
                        continue
                except (ValueError, TypeError, OverflowError):
                    pass

            if pd.api.types.is_integer_dtype(self.data[col]):
                self.data[col] = pd.to_numeric(self.data[col], downcast="integer")
            elif pd.api.types.is_float_dtype(self.data[col]):
                self.data[col] = pd.to_numeric(self.data[col], downcast="float")

        print("[convert_types] Type conversion complete.")
        return self.data

    # ──────────────────────────────────────────────────────────────
    # 9. Remove Duplicates
    # ──────────────────────────────────────────────────────────────
    def remove_duplicates(self) -> pd.DataFrame:
        """Remove exact duplicate rows."""
        before = len(self.data)
        self.data.drop_duplicates(inplace=True)
        removed = before - len(self.data)
        if removed:
            print(f"[remove_duplicates] Removed {removed} duplicate row(s).")
        return self.data

    # ──────────────────────────────────────────────────────────────
    # 10. Auto-Encode (User Mode — smart per‑column strategy)
    # ──────────────────────────────────────────────────────────────
    def auto_encode(self, onehot_max_unique: int = 10) -> pd.DataFrame:
        """
        Automatically encode ALL remaining categorical/object columns.

        Strategy:
            n_unique == 2          → Label encode  (0 / 1)
            3 ≤ n_unique ≤ limit   → One-Hot encode via sklearn OneHotEncoder
            n_unique > limit       → Label encode  (avoids column explosion)

        Args:
            onehot_max_unique : Upper bound for One-Hot encoding. Default 10.
        """
        cat_cols = self.data.select_dtypes(include=["object", "category"]).columns.tolist()

        if not cat_cols:
            print("[auto_encode] No categorical columns remaining.")
            return self.data

        onehot_cols = []
        label_cols = []

        for col in cat_cols:
            n_unique = self.data[col].nunique(dropna=True)

            if n_unique == 2:
                self.data[col] = self.data[col].astype("category").cat.codes
                label_cols.append(col)
                print(f"[auto_encode] Label-encoded binary '{col}' → 0/1.")

            elif n_unique <= onehot_max_unique:
                onehot_cols.append(col)

            else:
                self.data[col] = self.data[col].astype("category").cat.codes
                label_cols.append(col)
                print(f"[auto_encode] Label-encoded high-cardinality '{col}' ({n_unique} values).")

        if onehot_cols:
            encoder = OneHotEncoder(
                sparse_output=False,      # dense array
                handle_unknown="ignore",  # safe for future transforms
                dtype=int                 # output 0/1 integers
            )
            encoded_array = encoder.fit_transform(self.data[onehot_cols])
            new_col_names = encoder.get_feature_names_out(onehot_cols)

            encoded_df = pd.DataFrame(
                encoded_array,
                columns=new_col_names,
                index=self.data.index
            )

            self.data.drop(columns=onehot_cols, inplace=True)
            self.data = pd.concat([self.data, encoded_df], axis=1)

            print(f"[auto_encode] One-Hot encoded (sklearn, 0/1 int): {onehot_cols}")

        print(
            f"[auto_encode] Done — "
            f"{len(label_cols)} label-encoded, {len(onehot_cols)} one-hot encoded."
        )
        return self.data

    # ──────────────────────────────────────────────────────────────
    # 11. Encode Categoricals (Developer Mode — manual override)
    # ──────────────────────────────────────────────────────────────
    def encode_categoricals(self, method: str = "none") -> pd.DataFrame:
        """
        Manual encoding used in DEVELOPER MODE.
        Applies the same method to ALL remaining categorical columns.

        Args:
            method : 'none' | 'label' | 'onehot'
        """
        if method not in ("none", "label", "onehot"):
            method = "none"

        if method == "none":
            print("[encode_categoricals] Encoding skipped.")
            return self.data

        cat_cols = self.data.select_dtypes(include=["object", "category"]).columns.tolist()
        if not cat_cols:
            print("[encode_categoricals] No categorical columns to encode.")
            return self.data

        if method == "label":
            for col in cat_cols:
                self.data[col] = self.data[col].astype("category").cat.codes
            print(f"[encode_categoricals] Label-encoded: {cat_cols}")

        elif method == "onehot":
            encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
                dtype=int
            )
            encoded_array = encoder.fit_transform(self.data[cat_cols])
            new_col_names = encoder.get_feature_names_out(cat_cols)
            encoded_df = pd.DataFrame(
                encoded_array,
                columns=new_col_names,
                index=self.data.index
            )
            self.data.drop(columns=cat_cols, inplace=True)
            self.data = pd.concat([self.data, encoded_df], axis=1)
            print(f"[encode_categoricals] One-Hot encoded (sklearn, 0/1): {cat_cols}")

        return self.data

    # ──────────────────────────────────────────────────────────────
    # 12. Handle Outliers (delegates to OutlierHandler)
    # ──────────────────────────────────────────────────────────────
    def handle_outliers(
        self,
        method: str = "iqr",
        strategy: str = "cap",
        zscore_threshold: float = 3.0,
        original_schema: dict = None
    ) -> pd.DataFrame:
        """
        Delegates to OutlierHandler for IQR or Z‑score based detection and handling.

        Args:
            method          : 'iqr' or 'zscore'
            strategy        : 'cap' or 'remove'
            zscore_threshold: threshold for Z‑score method (default 3.0)
            original_schema : optional dict with original dtypes for reference
        """
        from Phase_1.OutlierHandler import OutlierHandler

        if method not in ("iqr", "zscore"):
            method = "iqr"
        if strategy not in ("cap", "remove"):
            strategy = "cap"

        handler = OutlierHandler(self.data, original_schema=original_schema)

        # Detect
        if method == "iqr":
            handler.detect_iqr()
        else:
            handler.detect_zscore(threshold=zscore_threshold)

        # Handle
        if strategy == "cap":
            self.data = handler.cap_outliers(method, zscore_threshold=zscore_threshold)
        else:
            self.data = handler.remove_outliers(method, zscore_threshold=zscore_threshold)

        print(f"[handle_outliers] Outliers handled — method={method}, strategy={strategy}.")
        return self.data

    # ──────────────────────────────────────────────────────────────
    # 13. Get Clean Data
    # ──────────────────────────────────────────────────────────────
    def get_clean_data(self) -> pd.DataFrame:
        """Return the fully processed DataFrame."""
        return self.data