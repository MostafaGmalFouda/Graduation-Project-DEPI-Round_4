import pandas as pd


class DataPreprocessor:
    """
    A class to preprocess data including handling nulls, converting types,
    removing duplicates, handling text columns, and handling outliers.

    Attributes:
        data (pd.DataFrame): The loaded data as a pandas DataFrame.

    Methods:
        handle_nulls(threshold, fill_strategy): Handle missing values.
        handle_text_columns(unique_threshold, action): Drop/hash/keep high-cardinality text columns.
        convert_types(schema): Convert column types.
        remove_duplicates(): Remove duplicate rows.
        exclude_columns(columns): Drop user-specified columns.
        encode_categoricals(method): Encode remaining categorical columns.
        get_clean_data(): Return the processed clean data as a DataFrame.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

    # ──────────────────────────────────────────────────────────────
    # 1. Handle Nulls
    # ──────────────────────────────────────────────────────────────
    def handle_nulls(self, threshold: float = 0.4, fill_strategy: str = "median") -> pd.DataFrame:
        """
        Automatically handles missing values:
        - If null ratio > threshold  → Drop the column.
        - If null ratio < 5%         → Drop the rows.
        - In between                 → Fill using fill_strategy for numeric,
                                       Mode for categorical.

        Args:
            threshold     : Ratio of nulls above which a column is dropped (default 0.4).
            fill_strategy : 'median' | 'mean' | 'mode' for numeric columns (default 'median').
        Returns:
            pd.DataFrame
        """
        if fill_strategy not in ("median", "mean", "mode"):
            fill_strategy = "median"

        for col in list(self.data.columns):          # list() → safe iteration while dropping
            null_count = self.data[col].isnull().sum()
            if null_count == 0:
                continue

            null_ratio = null_count / len(self.data)

            if null_ratio > threshold:
                print(f"[handle_nulls] Dropping column '{col}' — high null ratio ({null_ratio:.2%}).")
                self.data.drop(columns=[col], inplace=True)

            elif null_ratio < 0.05:
                print(f"[handle_nulls] Dropping rows with nulls in '{col}' ({null_ratio:.2%}).")
                self.data.dropna(subset=[col], inplace=True)

            else:
                if self.data[col].dtype in ['int64', 'float64']:
                    if fill_strategy == "mean":
                        fill_val = self.data[col].mean()
                    elif fill_strategy == "mode":
                        fill_val = self.data[col].mode()[0]
                    else:
                        fill_val = self.data[col].median()
                    # ✅ FIX: avoid deprecated inplace fillna
                    self.data[col] = self.data[col].fillna(fill_val)
                    print(f"[handle_nulls] Filled '{col}' with {fill_strategy}: {fill_val:.4g}.")
                else:
                    mode_val = self.data[col].mode()[0]
                    self.data[col] = self.data[col].fillna(mode_val)
                    print(f"[handle_nulls] Filled '{col}' with mode: '{mode_val}'.")

        return self.data

    # ──────────────────────────────────────────────────────────────
    # 2. Handle Text Columns  ← NEW
    # ──────────────────────────────────────────────────────────────
    def handle_text_columns(
        self,
        unique_threshold: float = 0.9,
        action: str = "drop"
    ) -> pd.DataFrame:
        """
        Detect and handle high-cardinality text (object) columns that are
        useless for modelling (e.g. Name, ID, free-text notes).

        Args:
            unique_threshold : If (n_unique / n_rows) >= this value the column
                               is considered 'ID-like'. Default 0.9 (90 %).
            action           : What to do with ID-like columns:
                               - 'drop'  → remove column entirely (default).
                               - 'hash'  → replace with numeric hash mod 10 000.
                               - 'keep'  → leave as-is (useful for inspection).
        Returns:
            pd.DataFrame
        """
        if action not in ("drop", "hash", "keep"):
            action = "drop"

        text_cols = self.data.select_dtypes(include=["object", "string"]).columns.tolist()

        if not text_cols:
            print("[handle_text_columns] No text columns found.")
            return self.data

        for col in text_cols:
            n_rows   = len(self.data)
            n_unique = self.data[col].nunique(dropna=True)

            if n_rows == 0:
                continue

            ratio = n_unique / n_rows

            if ratio >= unique_threshold:
                if action == "drop":
                    self.data.drop(columns=[col], inplace=True)
                    print(
                        f"[handle_text_columns] Dropped '{col}' — "
                        f"ID-like ({ratio:.0%} unique values)."
                    )
                elif action == "hash":
                    self.data[col] = self.data[col].apply(
                        lambda x: hash(str(x)) % 10_000
                    )
                    print(
                        f"[handle_text_columns] Hashed '{col}' → numeric "
                        f"({ratio:.0%} unique values)."
                    )
                else:  # keep
                    print(
                        f"[handle_text_columns] Keeping '{col}' as-is — "
                        f"ID-like ({ratio:.0%} unique values)."
                    )
            else:
                print(
                    f"[handle_text_columns] '{col}': {n_unique} unique values "
                    f"({ratio:.0%}) — will be handled by encode_categoricals."
                )

        return self.data

    # ──────────────────────────────────────────────────────────────
    # 3. Convert Types
    # ──────────────────────────────────────────────────────────────
    def convert_types(self, schema: dict = None) -> pd.DataFrame:
        """
        Converts columns to appropriate types.
        - Trims whitespace from string columns.
        - Applies schema-based conversion if provided.
        - Auto-detects datetime columns.
        - Downcasts numerics to save memory.

        Args:
            schema: Optional dict mapping column names to desired dtypes.
        Returns:
            pd.DataFrame
        """
        # 1. Trim whitespace from all object columns
        for col in self.data.select_dtypes(include=["object"]).columns:
            self.data[col] = self.data[col].astype(str).str.strip()

        # 2. Schema-based conversion (if provided)
        if schema:
            for col, dtype in schema.items():
                if col in self.data.columns:
                    try:
                        self.data[col] = self.data[col].astype(dtype)
                    except (ValueError, TypeError) as e:
                        print(f"[convert_types] Could not convert '{col}' to {dtype}: {e}")

        # 3. Auto-detect datetime & downcast numerics
        for col in self.data.columns:
            if self.data[col].dtype == "object":
                # ✅ FIX: catch only expected exceptions, not everything
                try:
                    converted = pd.to_datetime(self.data[col], infer_datetime_format=True)
                    # Only apply if at least 80% of values converted successfully
                    if converted.notna().mean() >= 0.8:
                        self.data[col] = converted
                        print(f"[convert_types] Converted '{col}' to datetime.")
                        continue
                except (ValueError, TypeError, OverflowError):
                    pass

            if pd.api.types.is_integer_dtype(self.data[col]):
                self.data[col] = pd.to_numeric(self.data[col], downcast="integer")
            elif pd.api.types.is_float_dtype(self.data[col]):
                self.data[col] = pd.to_numeric(self.data[col], downcast="float")

        print("[convert_types] Type conversion and memory optimisation completed.")
        return self.data

    # ──────────────────────────────────────────────────────────────
    # 4. Remove Duplicates
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
    # 5. Exclude Columns
    # ──────────────────────────────────────────────────────────────
    def exclude_columns(self, columns: list) -> pd.DataFrame:
        """
        Drop user-specified columns before any processing.
        Silently ignores names that do not exist.

        Args:
            columns: List of column names to drop.
        Returns:
            pd.DataFrame
        """
        if not columns:
            return self.data
        existing = [c for c in columns if c in self.data.columns]
        if existing:
            self.data.drop(columns=existing, inplace=True)
            print(f"[exclude_columns] Excluded {len(existing)} column(s): {existing}")
        return self.data

    # ──────────────────────────────────────────────────────────────
    # 6. Encode Categoricals
    # ──────────────────────────────────────────────────────────────
    def encode_categoricals(self, method: str = "none") -> pd.DataFrame:
        """
        Encode LOW-cardinality categorical (object) columns for modelling.
        High-cardinality text columns should already be handled by
        handle_text_columns() before calling this method.

        Args:
            method: 'none' | 'label' | 'onehot'
        Returns:
            pd.DataFrame
        """
        if method not in ("none", "label", "onehot"):
            method = "none"

        if method == "none":
            return self.data

        cat_cols = self.data.select_dtypes(include=["object", "category"]).columns.tolist()
        if not cat_cols:
            print("[encode_categoricals] No categorical columns to encode.")
            return self.data

        if method == "label":
            for col in cat_cols:
                self.data[col] = self.data[col].astype("category").cat.codes
            print(f"[encode_categoricals] Label-encoded {len(cat_cols)} column(s): {cat_cols}")

        elif method == "onehot":
            self.data = pd.get_dummies(self.data, columns=cat_cols)
            print(f"[encode_categoricals] One-hot encoded {len(cat_cols)} column(s): {cat_cols}")

        return self.data

    # ──────────────────────────────────────────────────────────────
    # 7. Handle Outliers (delegates to OutlierHandler)
    # ──────────────────────────────────────────────────────────────
    def handle_outliers(self) -> pd.DataFrame:
        """
        Delegates outlier handling to OutlierHandler (IQR capping by default).
        """
        from Phase_1.OutlierHandler import OutlierHandler
        handler = OutlierHandler(self.data)
        handler.detect_iqr()
        self.data = handler.cap_outliers("iqr")
        print("[handle_outliers] Outliers capped via IQR.")
        return self.data

    # ──────────────────────────────────────────────────────────────
    # 8. Get Clean Data
    # ──────────────────────────────────────────────────────────────
    def get_clean_data(self) -> pd.DataFrame:
        """Return the processed DataFrame."""
        return self.data