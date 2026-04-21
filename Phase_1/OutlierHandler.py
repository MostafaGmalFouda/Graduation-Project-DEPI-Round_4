import pandas as pd


class OutlierHandler:
    """
    A class to detect and handle outliers in a dataset.

    Attributes:
        data (pd.DataFrame): The dataset as a pandas DataFrame.

    Methods:
        detect_iqr(): Detects outliers using IQR method.
        detect_zscore(): Detects outliers using Z-score method.
        remove_outliers(): Removes rows containing outliers.
        cap_outliers(): Caps outliers to boundary values.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data


    def detect_iqr(self) -> dict:
        """
        Detect outliers using the IQR method.

        Outliers are values outside:
        [Q1 - 1.5*IQR , Q3 + 1.5*IQR]

        Returns:
            dict: Number of outliers in each column.
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

        Returns:
            dict: Number of outliers in each column.
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


def remove_outliers(self, method: str = 'iqr') -> pd.DataFrame:
    """
    Remove rows that contain outliers using selected method.

    Parameters:
        method (str): Detection method ('iqr' or 'zscore').

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    try:
        numeric_cols = self.data.select_dtypes(include='number').columns
        clean_data = self.data.copy()

        for col in numeric_cols:

            # IQR METHOD
            if method.lower() == 'iqr':
                Q1 = clean_data[col].quantile(0.25)
                Q3 = clean_data[col].quantile(0.75)
                IQR = Q3 - Q1

                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                clean_data = clean_data[
                    (clean_data[col] >= lower) & (clean_data[col] <= upper)
                ]

            # Z-SCORE METHOD
            elif method.lower() == 'zscore':
                mean = clean_data[col].mean()
                std = clean_data[col].std()

                if std == 0:
                    continue

                z_scores = (clean_data[col] - mean) / std

                clean_data = clean_data[
                    abs(z_scores) <= 3
                ]

            else:
                raise ValueError("Method must be 'iqr' or 'zscore'")

        print(f"Outliers removed successfully using {method.upper()} method.")
        return clean_data

    except Exception as e:
        raise Exception(f"Error removing outliers: {e}")


def cap_outliers(self, lower_bound=None, upper_bound=None) -> pd.DataFrame:
    """
    Cap all numeric values within a given range.

    If bounds are not provided, they will be calculated
    automatically using global percentiles (5% - 95%).

    Parameters:
        lower_bound (float, optional): Minimum allowed value.
        upper_bound (float, optional): Maximum allowed value.

    Returns:
        pd.DataFrame: Capped dataset.
    """
    try:
        capped_data = self.data.copy()

        numeric_cols = capped_data.select_dtypes(include='number').columns

        if lower_bound is None:
            lower_bound = capped_data[numeric_cols].quantile(0.05).min()

        if upper_bound is None:
            upper_bound = capped_data[numeric_cols].quantile(0.95).max()

        for col in numeric_cols:
            capped_data[col] = capped_data[col].clip(lower_bound, upper_bound)

        print(f"Data capped between [{lower_bound}, {upper_bound}] successfully.")
        return capped_data

    except Exception as e:
        raise Exception(f"Error in cap_outliers: {e}")