import pandas as pd

class DataPreprocessor:
    """
    A class to preprocess data including handling nulls, converting types,
    removing duplicates, and handling outliers.
    Attributes:      
        data (pd.DataFrame): The loaded data as a pandas DataFrame.
    Methods:       
        handle_nulls(strategy): Handle missing values based on the specified strategy.
        convert_types(schema): Convert column types based on a provided schema.
        remove_duplicates(): Remove duplicate rows from the data.
        handle_outliers(method): Handle outliers using specified method (IQR or Z-score).
        get_clean_data(): Return the processed clean data as a DataFrame.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()


    def handle_nulls(self, threshold :float = 0.4) -> pd.DataFrame:
        """
        Automatically handles missing values:
        - If null ratio > threshold (40%): Drop the column.
        - If null ratio < 5%: Drop the rows.
        - In between: Fill with Median (Numeric) or Mode (Categorical).
        threshold: The ratio of nulls above which a column will be dropped.
        Returns:
            pd.DataFrame: The DataFrame with nulls handled.
        """
        for col in self.data.columns:
            null_count = self.data[col].isnull().sum()
            if null_count == 0:
                continue
            
            null_ratio = null_count / len(self.data)
            
            if null_ratio > threshold:
                print(f"Dropping column '{col}' due to high null ratio ({null_ratio:.2%}).")
                self.data.drop(columns=[col], inplace=True)
            
            elif null_ratio < 0.05:
                print(f"Dropping rows with nulls in '{col}' (low ratio: {null_ratio:.2%}).")
                self.data.dropna(subset=[col], inplace=True)
            
            else:
                # Fill with Median for numeric, Mode for categorical
                if self.data[col].dtype in ['int64', 'float64']:
                    median_val = self.data[col].median()
                    self.data[col].fillna(median_val, inplace=True)
                    print(f"Filled nulls in '{col}' with Median: {median_val}.")
                else:
                    mode_val = self.data[col].mode()[0]
                    self.data[col].fillna(mode_val, inplace=True)
                    print(f"Filled nulls in '{col}' with Mode: {mode_val}.")
        
        return self.data


    # def handle_nulls(self, strategy="mean") -> pd.DataFrame:
    #     """
    #     Handle missing values based on strategy.
    #     strategy: mean, median, mode, drop
    #     """
    #     if strategy == "mean":
    #             numeric_cols = self.data.select_dtypes(include=['number']).columns
    #             self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].mean())
    #             print("Null values filled with mean.")      
    #     elif strategy == "median":
    #             numeric_cols = self.data.select_dtypes(include=['number']).columns
    #             self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].median())
    #             print("Null values filled with median.")
    #     elif strategy == "mode":
    #         self.data = self.data.fillna(self.data.mode().iloc[0])
    #         print("Null values filled with mode.")
    #     elif strategy == "drop":
    #         self.data = self.data.dropna()
    #         print("Rows with null values dropped.")
    #     else:
    #         raise ValueError("Invalid null handling strategy")

    #     return self.data
    

    def convert_types(self, schema: dict = None) -> pd.DataFrame:
        """
        Converts columns to appropriate types. If no schema is provided,
        it tries to infer types automatically (like converting strings to dates).
        schema: A dictionary mapping column names to desired data types (e.g., {'date_col': 'datetime', 'num_col': 'float'}).
        """
        # 1. cleaning string columns (trimming whitespace)
        for col in self.data.select_dtypes(include=['object']).columns:
            self.data[col] = self.data[col].astype(str).str.strip()

        # 2. schema-based conversion
        if schema:
            for col, dtype in schema.items():
                if col in self.data.columns:
                    self.data[col] = self.data[col].astype(dtype)
        
        # 3. automatic conversion (simple AI)
        for col in self.data.columns:
            # convert date-like strings to datetime
            if self.data[col].dtype == 'object':
                try:
                    self.data[col] = pd.to_datetime(self.data[col])
                    continue 
                except:
                    pass
            
            # reduce memory usage (Downcasting) for numbers
            if self.data[col].dtype in ['int64', 'float64']:
                self.data[col] = pd.to_numeric(self.data[col], downcast='integer' if 'int' in str(self.data[col].dtype) else 'float')

        print("Advanced type conversion and memory optimization completed.")
        return self.data


    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate rows
        """
        initial_count = len(self.data)
        self.data.drop_duplicates(inplace=True)
        final_count = len(self.data)
        if initial_count != final_count:
            print(f"Removed {initial_count - final_count} duplicate rows.")
        return self.data
    

    def handle_outliers(self) -> pd.DataFrame:
        """
        Delegates outlier handling to the OutlierHandler class.
        Uses the 'Capping' method by default as it's the safest for automation.
        """
        from OutlierHandler import OutlierHandler # Importing class here to avoid circular dependency if OutlierHandler also imports DataPreprocessor
        
        handler = OutlierHandler(self.data)
        self.data = handler.cap_outliers() # Capping (IQR)
        print("Outliers handled automatically via OutlierHandler.")
        return self.data


    # def handle_outliers(self, method="iqr") -> pd.DataFrame:
    #     """
    #     Handle outliers using IQR or Z-score
    #     """
    #     if method == "iqr":
    #         Q1 = self.data.quantile(0.25)
    #         Q3 = self.data.quantile(0.75)
    #         IQR = Q3 - Q1

    #         self.data = self.data[~((self.data < (Q1 - 1.5 * IQR)) | 
    #                                 (self.data > (Q3 + 1.5 * IQR))).any(axis=1)]

    #     elif method == "zscore":
    #         from scipy import stats
    #         z_scores = stats.zscore(self.data.select_dtypes(include=['number']))
    #         self.data = self.data[(abs(z_scores) < 3).all(axis=1)]

    #     else:
    #         raise ValueError("Invalid outlier handling method")

    #     return self.data


    def get_clean_data(self) -> pd.DataFrame:
        """
        Return the processed clean data
        """
        return self.data