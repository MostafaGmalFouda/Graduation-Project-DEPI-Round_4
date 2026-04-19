import pandas as pd

class DataValidator:
    """
    A class to validate the loaded data for consistency, completeness, and correctness.
    Attributes:
        data (pd.DataFrame): The loaded data as a pandas DataFrame.
    Methods:
        check_nulls(): Checks for missing values in the data.
        check_types(): Checks for correct data types in the columns.
        check_duplicates(): Checks for duplicate rows in the data.
        report_issues():Returns a  dict of all validation issues.
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data


    def check_nulls(self) -> dict:
        """
        Checks for missing values in the data.
        Returns:
            dict: of columns with missing values and their counts.
        """
        try:
            null_counts = self.data.isnull().sum()
            result = null_counts[null_counts > 0].to_dict()
            if result:
                print(f"Null values found in {len(result)} column(s).")
            else:
                print("No null values found.")
            return result
        except Exception as e:
            raise Exception(f"Error checking nulls: {e}")
        
    def check_types(self) -> dict:
        """
        Retrieves the data type of each column in the DataFrame.

        Returns:
            dict: A dictionary mapping column names to their dtype as a string.
        """
        try:
            types = {col: str(dtype) for col, dtype in self.data.dtypes.items()}
            print("Data types retrieved successfully.")
            return types
        except Exception as e:
            raise Exception(f"Error checking types: {e}")
        

    def check_duplicates(self) -> int:
        """
        Counts the number of fully duplicate rows in the DataFrame.

        Returns:
            int: The number of duplicate rows found.
        """
        try:
            duplicate_count = int(self.data.duplicated().sum())
            if duplicate_count > 0:
                print(f"{duplicate_count} duplicate row(s) found.")
            else:
                print("No duplicate rows found.")
            return duplicate_count
        except Exception as e:
            raise Exception(f"Error checking duplicates: {e}")
        
   

    def report_issues(self) -> dict:
        """
        Validates the data and returns a report of any issues found.
        Returns:
            dict: A report containing any issues found during validation.
        """
        report = {
            "missing_values": self.check_nulls(),
            "data_types": self.check_types(),
            "duplicates": self.check_duplicates()
        }
        return report