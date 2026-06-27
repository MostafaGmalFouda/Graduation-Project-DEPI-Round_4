import pandas as pd
from sklearn.preprocessing import OneHotEncoder
class DataPreprocessor:
    """
    Smart preprocessing pipeline for tabular data.

    Steps (in order):
        1. exclude_columns       — drop user-specified columns
        2. drop_id_columns         — drop numeric ID-like columns (NEW)
        3. handle_nulls          — drop/fill missing values
        4. handle_text_columns   — drop/hash ID-like text columns
        5. convert_types         — trim, datetime detection, numeric downcast
        6. remove_duplicates     — exact-row deduplication
        7. auto_encode           — smart encoding for remaining categoricals (USER MODE)
        8. encode_categoricals   — manual encoding override (DEVELOPER MODE)
        9. Handle_outliers       ـ handle outliers
        10. get_clean_data        — return the processed DataFrame
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
    # 2. Drop Numeric ID Columns 
    # ──────────────────────────────────────────────────────────────
    # def drop_id_columns(self) -> pd.DataFrame:
    #     """
    #     Detect and drop numeric columns that are just row identifiers
    #     (e.g. PassengerId, CustomerID, RowNum).
 
    #     A numeric column is considered an ID if ALL of:
    #         - All values are unique (n_unique == n_rows)
    #         - Values are sequential integers (max - min + 1 == n_rows)
    #         - Column name contains a known ID keyword (case-insensitive)
    #           OR the above two conditions alone are strong enough.
    #     """
    #     id_keywords = {"id", "index", "rownum", "row_num", "passengerId",
    #                    "customerid", "userid", "number", "num", "no", "code"}
 
    #     numeric_cols = self.data.select_dtypes(include=["number"]).columns.tolist()
    #     dropped = []
 
    #     for col in numeric_cols:
    #         n_rows   = len(self.data)
    #         n_unique = self.data[col].nunique()
 
    #         all_unique  = (n_unique == n_rows)
    #         col_lower   = col.lower().replace("_", "").replace(" ", "")
    #         has_keyword = any(kw in col_lower for kw in id_keywords)
 
    #         # Sequential integer check
    #         try:
    #             col_min = self.data[col].min()
    #             col_max = self.data[col].max()
    #             is_sequential = (col_max - col_min + 1) == n_rows
    #         except Exception:
    #             is_sequential = False
 
    #         if all_unique and (has_keyword or is_sequential):
    #             self.data.drop(columns=[col], inplace=True)
    #             dropped.append(col)
    #             print(f"[drop_id_columns] Dropped numeric ID column '{col}' "
    #                   f"({n_unique}/{n_rows} unique, sequential={is_sequential}).")
 
    #     if not dropped:
    #         print("[drop_id_columns] No numeric ID columns detected.")
 
    #     return self.data
    
    # ──────────────────────────────────────────────────────────────
    # 3. Handle Nulls
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
                    mode_val = self.data[col].mode()[0]
                    self.data[col] = self.data[col].fillna(mode_val)
                    print(f"[handle_nulls] Filled '{col}' with mode: '{mode_val}'.")

        return self.data

    # ──────────────────────────────────────────────────────────────
    # 4. Handle Text Columns  (ID-like / free-text)
    # ──────────────────────────────────────────────────────────────
    def handle_text_columns(
        self,
        unique_threshold: float = 0.6,
        action: str = "drop"
    ) -> pd.DataFrame:
        """
        Detect and handle high-cardinality text columns useless for modelling
        (e.g. Name, Ticket, PassengerId).

        Args:
            unique_threshold : Columns where n_unique/n_rows >= this → ID-like. Default 0.9.
            action           : 'drop' | 'hash' | 'keep'
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
            
            is_mixed_alphanum = self._is_mixed_alphanum(col)
            
            if ratio >= unique_threshold or is_mixed_alphanum:
                if action == "drop":
                    self.data.drop(columns=[col], inplace=True)
                    reason = f"mixed alphanumeric" if is_mixed_alphanum else f"{ratio:.0%} unique"
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
        """
        Returns True if the column looks like a ticket/ID column:
        contains both letters AND digits in most of its values.
        """
        try:
            sample = self.data[col].dropna().astype(str).head(50)
            mixed_count = sample.apply(
                lambda x: bool(any(c.isalpha() for c in x) and any(c.isdigit() for c in x))
            ).sum()
            return (mixed_count / len(sample)) >= 0.3 if len(sample) > 0 else False
        except Exception:
            return False

    # ──────────────────────────────────────────────────────────────
    # 5. Convert Types
    # ──────────────────────────────────────────────────────────────
    def convert_types(self, schema: dict = None) -> pd.DataFrame:
        """
        - Trim whitespace from object columns.
        - Schema-based conversion if provided.
        - Auto-detect datetime strings.
        - Downcast numerics to save memory.
        """
        for col in self.data.select_dtypes(include=["object"]).columns:
            self.data[col] = self.data[col].astype(str).str.strip()

        if schema:
            for col, dtype in schema.items():
                if col in self.data.columns:
                    try:
                        self.data[col] = self.data[col].astype(dtype)
                    except (ValueError, TypeError) as e:
                        print(f"[convert_types] Cannot convert '{col}' to {dtype}: {e}")

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
    # 6. Remove Duplicates
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
    # 7. Auto-Encode 
    # ──────────────────────────────────────────────────────────────
    def auto_encode(self, onehot_max_unique: int = 10) -> pd.DataFrame:
        """
        Automatically encode ALL remaining categorical/object columns.
 
        Strategy:
            Binary  (2 unique)          → Label encode  → 0 / 1
            Low-cardinality (≤ limit)   → One-Hot encode → integer 0 / 1  ✅ FIX: dtype=int
            High-cardinality (> limit)  → Label encode
 
        Args:
            onehot_max_unique : Max unique values for One-Hot. Default 10.
        """
        cat_cols = self.data.select_dtypes(include=["object", "category"]).columns.tolist()
 
        if not cat_cols:
            print("[auto_encode] No categorical columns remaining.")
            return self.data
 
        onehot_cols = []
        label_cols  = []
 
        for col in cat_cols:
            n_unique = self.data[col].nunique(dropna=True)
 
            if n_unique == 2:
                self.data[col] = self.data[col].astype("category").cat.codes
                label_cols.append(col)
                print(f"[auto_encode] Label-encoded binary '{col}' ({n_unique} values) → 0/1.")
 
            elif n_unique <= onehot_max_unique:
                onehot_cols.append(col)
 
            else:
                self.data[col] = self.data[col].astype("category").cat.codes
                label_cols.append(col)
                print(f"[auto_encode] Label-encoded high-cardinality '{col}' ({n_unique} values).")
        
        if onehot_cols:
            self.data = pd.get_dummies(self.data, columns=onehot_cols, drop_first=False, dtype=int)
            print(f"[auto_encode] One-Hot encoded (0/1 integers): {onehot_cols}")
 
        print(
            f"[auto_encode] Done — "
            f"{len(label_cols)} label-encoded, {len(onehot_cols)} one-hot encoded."
        )
        return self.data

    # ──────────────────────────────────────────────────────────────
    # 8. Encode Categoricals  (Developer Mode manual override)
    # ──────────────────────────────────────────────────────────────
    def encode_categoricals(self, method: str = "none") -> pd.DataFrame:
        """
        Manual encoding used in DEVELOPER MODE.
        Applies the same method to ALL remaining categorical columns.

        Args:
            method: 'none' | 'label' | 'onehot'
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
            self.data = pd.get_dummies(self.data, columns=cat_cols, drop_first=False)
            print(f"[encode_categoricals] One-Hot encoded: {cat_cols}")

        return self.data

    # ──────────────────────────────────────────────────────────────
    # 9. Handle Outliers (delegates to OutlierHandler)
    # ──────────────────────────────────────────────────────────────
    def handle_outliers(self) -> pd.DataFrame:
        """Delegates to OutlierHandler — IQR capping by default."""
        from Phase_1.OutlierHandler import OutlierHandler
        handler = OutlierHandler(self.data)
        handler.detect_iqr()
        self.data = handler.cap_outliers("iqr")
        print("[handle_outliers] Outliers capped via IQR.")
        return self.data

    # ──────────────────────────────────────────────────────────────
    # 10. Get Clean Data
    # ──────────────────────────────────────────────────────────────
    def get_clean_data(self) -> pd.DataFrame:
        """Return the fully processed DataFrame."""
        return self.data