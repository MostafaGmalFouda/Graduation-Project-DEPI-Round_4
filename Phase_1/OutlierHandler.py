import pandas as pd


class OutlierHandler:
    """
    A class to detect and handle outliers in a dataset.

    Attributes:
        data (pd.DataFrame): The dataset as a pandas DataFrame.

    Methods:
        detect_iqr()                        : Detects outliers using IQR method.
        detect_zscore(threshold)            : Detects outliers using Z-score method.
        remove_outliers(method)             : Removes rows containing outliers.
        cap_outliers(method)                : Caps outliers to boundary values.
    """

    def __init__(self, data: pd.DataFrame, original_schema: dict = None):
        """
        original_schema: {col_name: 'num'|'cat'} computed on the RAW data
        BEFORE any encoding. None → falls back to old dtype-only behavior.
        """
        self.data = data
        self.original_schema = original_schema


    def _numeric_non_binary_cols(self, discrete_threshold: int = 10) -> list:
        """
        Columns eligible for outlier detection = genuinely CONTINUOUS numeric
        measurements only (Age, Fare...). Excludes:
        - columns that weren't numeric before encoding (Sex/Name/Ticket after
            label-encoding, One-Hot indicators) — via original_schema
        - binary 0/1 columns
        - ID-like columns (~unique per row, e.g. PassengerId)
        - low-cardinality DISCRETE numeric columns (SibSp, Parch, Pclass) —
            counts/categories, not measurements. Capping these can even
            collapse a column to a constant (Parch: Q1==Q3==0).
        """
        numeric_cols = self.data.select_dtypes(include='number').columns
        safe_cols = []
        n_rows = len(self.data)

        for col in numeric_cols:
            if self.original_schema is not None and self.original_schema.get(col, 'cat') != 'num':
                continue

            series = self.data[col].dropna()
            if series.empty:
                continue

            unique_vals = series.unique()
            n_unique = len(unique_vals)

            if set(unique_vals).issubset({0, 1}):             # binary indicator
                continue
            if n_rows > 0 and (n_unique / n_rows) >= 0.9:     # ID-like
                continue
            if n_unique <= discrete_threshold:                # discrete/count/category
                continue

            safe_cols.append(col)

        return safe_cols

    # ──────────────────────────────────────────────────────────────
    # Detect — IQR
    # ──────────────────────────────────────────────────────────────
    def detect_iqr(self) -> dict:
        """
        Detect outliers using the IQR method.
        Skips binary (0/1) columns — they have no meaningful outliers.

        Returns:
            dict: Number of outliers per column.
        """
        try:
            outliers = {}
            for col in self._numeric_non_binary_cols():
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                count = ((self.data[col] < lower) | (self.data[col] > upper)).sum()
                outliers[col] = int(count)

            print("IQR outlier detection completed.")
            return outliers

        except Exception as e:
            raise Exception(f"Error in IQR detection: {e}")

    # ──────────────────────────────────────────────────────────────
    # Detect — Z-Score
    # ──────────────────────────────────────────────────────────────
    def detect_zscore(self, threshold: float = 3.0) -> dict:
        """
        Detect outliers using Z-score method.
        Skips binary (0/1) columns.

        Args:
            threshold: Z-score cutoff. Defaults to 3.0.
        Returns:
            dict: Number of outliers per column.
        """
        try:
            outliers = {}
            for col in self._numeric_non_binary_cols():
                mean = self.data[col].mean()
                std  = self.data[col].std()
                if std == 0:
                    outliers[col] = 0
                    continue
                z_scores = (self.data[col] - mean) / std
                count = (abs(z_scores) > threshold).sum()
                outliers[col] = int(count)

            print("Z-score outlier detection completed.")
            return outliers

        except Exception as e:
            raise Exception(f"Error in Z-score detection: {e}")

    # ──────────────────────────────────────────────────────────────
    # Remove
    # ──────────────────────────────────────────────────────────────
    def remove_outliers(self, method: str = "iqr", zscore_threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove rows that contain outliers.
        Skips binary (0/1) columns.

        Args:
            method            : 'iqr' or 'zscore'. Defaults to 'iqr'.
            zscore_threshold  : Cutoff when method='zscore'. Defaults to 3.0.
        Returns:
            pd.DataFrame: Cleaned dataset with outlier rows removed.
        """
        try:
            if method not in ('iqr', 'zscore'):
                raise ValueError("Invalid method. Choose 'iqr' or 'zscore'.")

            clean_data = self.data.copy()

            for col in self._numeric_non_binary_cols():
                if col not in clean_data.columns:
                    continue

                if method == 'iqr':
                    Q1 = clean_data[col].quantile(0.25)
                    Q3 = clean_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    clean_data = clean_data[
                        (clean_data[col] >= lower) & (clean_data[col] <= upper)
                    ]

                elif method == 'zscore':
                    mean = clean_data[col].mean()
                    std  = clean_data[col].std()
                    if std == 0:
                        continue
                    z_scores = (clean_data[col] - mean) / std
                    clean_data = clean_data[abs(z_scores) <= zscore_threshold]

            print(f"Outliers removed successfully using {method.upper()} method.")
            return clean_data

        except Exception as e:
            raise Exception(f"Error removing outliers: {e}")

    # ──────────────────────────────────────────────────────────────
    # Cap
    # ──────────────────────────────────────────────────────────────
    def cap_outliers(self, method: str = "iqr", zscore_threshold: float = 3.0) -> pd.DataFrame:
        """
        Cap outliers to boundary values instead of removing them.
        Skips binary (0/1) columns — capping them would wipe the encoding.

        Args:
            method            : 'iqr' or 'zscore'. Defaults to 'iqr'.
            zscore_threshold  : Cutoff when method='zscore'. Defaults to 3.0.
        Returns:
            pd.DataFrame: Dataset with outliers capped.
        """
        try:
            if method not in ('iqr', 'zscore'):
                raise ValueError("Invalid method. Choose 'iqr' or 'zscore'.")

            capped_data = self.data.copy()

            for col in self._numeric_non_binary_cols():
                if col not in capped_data.columns:
                    continue

                # ✅ FIX: initialise bounds to None so Pylint knows they are
                # always defined before use, even if method is unexpected.
                lower_bound = None
                upper_bound = None

                if method == 'iqr':
                    Q1 = capped_data[col].quantile(0.25)
                    Q3 = capped_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                elif method == 'zscore':
                    mean = capped_data[col].mean()
                    std  = capped_data[col].std()
                    if std == 0:
                        continue
                    lower_bound = mean - zscore_threshold * std
                    upper_bound = mean + zscore_threshold * std

                # Skip if bounds were never set (should not happen after
                # validation above, but guards against future edge cases)
                if lower_bound is None or upper_bound is None:
                    continue

                capped_data[col] = capped_data[col].clip(lower_bound, upper_bound)
                print(f"  '{col}' capped between [{round(lower_bound, 2)}, {round(upper_bound, 2)}]")

            print(f"\nData capped successfully using {method.upper()} method.")
            return capped_data

        except Exception as e:
            raise Exception(f"Error in cap_outliers: {e}")