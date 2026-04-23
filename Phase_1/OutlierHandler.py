import pandas as pd


class OutlierHandler:
    """
    A class to detect and handle outliers in a dataset.

    Attributes:
        data (pd.DataFrame): The dataset as a pandas DataFrame.

    Methods:
        detect_iqr()                        : Detects outliers using IQR method.
        detect_zscore()                     : Detects outliers using Z-score method.
        remove_outliers(method)             : Removes rows containing outliers.
        cap_outliers(method)                : Caps outliers to boundary values.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data


    def detect_iqr(self) -> dict:
        """
        Detect outliers using the IQR method.

        Outliers are values outside:
        [Q1 - 1.5*IQR , Q3 + 1.5*IQR]

        Returns:
            dict: Number of outliers per column.
        """
        try:
            outliers = {}
            numeric_cols = self.data.select_dtypes(include='number').columns

            for col in numeric_cols:
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


    def detect_zscore(self) -> dict:
        """
        Detect outliers using Z-score method.

        Z = (x - mean) / std
        Outliers are values where |Z| > 3.

        Returns:
            dict: Number of outliers per column.
        """
        try:
            outliers = {}
            numeric_cols = self.data.select_dtypes(include='number').columns

            for col in numeric_cols:
                mean = self.data[col].mean()
                std = self.data[col].std()

                if std == 0:
                    outliers[col] = 0
                    continue

                z_scores = (self.data[col] - mean) / std
                count = (abs(z_scores) > 3).sum()
                outliers[col] = int(count)

            print("Z-score outlier detection completed.")
            return outliers

        except Exception as e:
            raise Exception(f"Error in Z-score detection: {e}")


    def remove_outliers(self, method: str = "iqr") -> pd.DataFrame:
        """
        Remove rows that contain outliers.

        Args:
            method (str): 'iqr' or 'zscore'. Defaults to 'iqr'.

        Returns:
            pd.DataFrame: Cleaned dataset with outlier rows removed.

        Raises:
            ValueError: If method is not 'iqr' or 'zscore'.
        """
        try:
            if method not in ('iqr', 'zscore'):
                raise ValueError("Invalid method. Choose 'iqr' or 'zscore'.")

            numeric_cols = self.data.select_dtypes(include='number').columns
            clean_data = self.data.copy()

            for col in numeric_cols:

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
                    std = clean_data[col].std()
                    if std == 0:
                        continue
                    z_scores = (clean_data[col] - mean) / std
                    clean_data = clean_data[abs(z_scores) <= 3]

            print(f"Outliers removed successfully using {method.upper()} method.")
            return clean_data

        except Exception as e:
            raise Exception(f"Error removing outliers: {e}")


    def cap_outliers(self, method: str = "iqr") -> pd.DataFrame:
        """
        Cap outliers to boundary values instead of removing them.
        Capping is safer than removal — it keeps all rows and just
        clips extreme values to the boundary.

        Args:
            method (str): 'iqr' or 'zscore'. Defaults to 'iqr'.

        Returns:
            pd.DataFrame: Dataset with outliers capped.

        Raises:
            ValueError: If method is not 'iqr' or 'zscore'.
        """
        try:
            if method not in ('iqr', 'zscore'):
                raise ValueError("Invalid method. Choose 'iqr' or 'zscore'.")

            capped_data = self.data.copy()
            numeric_cols = capped_data.select_dtypes(include='number').columns

            for col in numeric_cols:

                if method == 'iqr':
                    Q1 = capped_data[col].quantile(0.25)
                    Q3 = capped_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                elif method == 'zscore':
                    mean = capped_data[col].mean()
                    std = capped_data[col].std()
                    if std == 0:
                        continue
                    lower_bound = mean - 3 * std
                    upper_bound = mean + 3 * std

                capped_data[col] = capped_data[col].clip(lower_bound, upper_bound)
                print(f"  '{col}' capped between [{round(lower_bound, 2)}, {round(upper_bound, 2)}]")

            print(f"\nData capped successfully using {method.upper()} method.")
            return capped_data

        except Exception as e:
            raise Exception(f"Error in cap_outliers: {e}")