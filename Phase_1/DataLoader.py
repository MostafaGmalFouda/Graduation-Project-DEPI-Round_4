import os
import pandas as pd

class DataLoader:
    """
    DataLoader Class

    Responsible for loading datasets from:
    - CSV files
    - Excel files

    Attributes:
        file_path (str): Path to the dataset
        data (pd.DataFrame): Loaded data

    Methods:
        load(): Detect file type and load data
        load_csv(): Load CSV file
        load_excel(): Load Excel file
        get_data(): Return loaded data
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load(self) -> pd.DataFrame:
        """
        Detect file type and load data.

        Returns:
            pd.DataFrame
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError("File not found")

        if self.file_path.endswith(".csv"):
            return self.load_csv()

        elif self.file_path.endswith((".xls", ".xlsx")):
            return self.load_excel()

        else:
            raise ValueError("Unsupported file format")

    def load_csv(self) -> pd.DataFrame:
        """
        Load CSV file.

        Returns:
            pd.DataFrame
        """
        try:
            self.data = pd.read_csv(self.file_path)
            return self.data
        except Exception as e:
            raise Exception(f"CSV Error: {e}")

    def load_excel(self) -> pd.DataFrame:
        """
        Load Excel file.

        Returns:
            pd.DataFrame
        """
        try:
            self.data = pd.read_excel(self.file_path)
            return self.data
        except Exception as e:
            raise Exception(f"Excel Error: {e}")

    def get_data(self) -> pd.DataFrame:
        """
        Return loaded data.

        Returns:
            pd.DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded yet")
        return self.data