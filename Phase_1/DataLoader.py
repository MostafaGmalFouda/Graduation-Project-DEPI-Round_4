import os
import pandas as pd

class DataLoader:
    """
    A class to load data from various file formats (CSV, Excel) into a pandas DataFrame.
    Attributes:
        file_path (str): The path to the data file.
        data (pd.DataFrame): The loaded data as a pandas DataFrame.
    Methods:
        load(): Loads the data from the specified file path and returns a DataFrame.
        load_csv(): Loads data from a CSV file.
        load_excel(): Loads data from an Excel file.
        get_data(): Returns the loaded data as a DataFrame.
    """
    def __init__(self,file_path):
        self.file_path = file_path
        self.data = None

    def load(self) ->pd.DataFrame:
        """
        Loads the data from the specified file path and returns a DataFrame.
        Autmatically detects the file format based on the file extension and calls the appropriate loading method.
        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File {self.file_path} does not exist.")
        
        if self.file_path.endswith(".csv"):
            return self.load_csv()
        
        elif self.file_path.endswith((".xls", ".xlsx")):
            return self.load_excel()
        
        else:
            raise ValueError("Unsupported file format. Only .csv, .xls, and .xlsx are supported.")
        
    def load_csv(self) -> pd.DataFrame:
        """
        Loads data from a CSV file.
        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        try:
            self.data = pd.read_csv(self.file_path)
            print("CSV file loaded successfully")
        except Exception as e:
            raise Exception(f"Error loading CSV: {e}")
        
    def load_excel(self) -> pd.DataFrame:
        """
        Loads data from an Excel file.
        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        try:
            self.data = pd.read_excel(self.file_path)
            print("Excel file loaded successfully")
        except Exception as e:
            raise Exception(f"Error loading Excel: {e}")
        
    def get_data(self) -> pd.DataFrame:
        """
        Returns the loaded data as a DataFrame.
        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        if self.data is None:
            raise ValueError("No data loaded yet. Call load() first.")
        else:
            return self.data
    
    