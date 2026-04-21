import pandas as pd
from ydata_profiling import ProfileReport

class  ReportGenerator:
   """
    A class for generating Exploratory Data Analysis (EDA) reports.

    This class supports two modes of report generation:
    1. Manual mode: basic statistical and structural analysis.
    2. Auto mode: full automated profiling report using ydata_profiling.

    Attributes:
        data (pd.DataFrame): The dataset used for analysis.
    """
   def __init__(self,data:pd.DataFrame):
        """
        Initialize the ReportGenerator with a dataset.

        Args:
            data (pd.DataFrame): The input dataset for analysis.
        """
        self.data =data

   def summary_stats(self):
       """
        Compute descriptive statistics for numeric columns.

        Returns:
            pd.DataFrame: Summary statistics (mean, std, min, max, etc.).
       """
       return self.data.describe()
   

   def correlation_matrix(self):
        """
        Compute correlation matrix for numeric features.

        Returns:
            pd.DataFrame: Correlation matrix between numeric columns.
        """
        return self.data.corr(numeric_only=True)
   
   def missing_values(self):
        """
        Calculate missing values for each column.

        Returns:
            pd.Series: Count of missing values per column.
        """
        return self.data.isnull().sum()
   

   def manual_report(self):
        """
        Generate a manual EDA report containing basic analysis.

        The report includes:
        - Summary statistics
        - Correlation matrix
        - Missing values analysis

        Returns:
            dict: A dictionary containing DataFrames of analysis results.
        """
        report ={
            "Summary":self.summary_stats(),
             "Correlation":self.correlation_matrix(),
             "Missing Values":self.missing_values()
        }
        return report


   def auto_report(self, file_name="report.html", explorative=True):
        """
        Generate an automated EDA report using ydata_profiling.

        Args:
            file_name (str): Output HTML file name.
            explorative (bool): If True, enables deep exploratory analysis.

        Returns:
            str: Confirmation message with saved file name.
        """
        profile = ProfileReport(self.data,explorative=explorative)
        profile.to_file(file_name)
        return f"Auto report saved as {file_name}"
            
   def generate_report(self, mode="manual", file_name="report.html"):
       """
        Generate an EDA report based on selected mode.

        Modes:
            - "manual": returns structured analysis as Python objects.
            - "auto": generates full HTML profiling report.

        Args:
            mode (str): Type of report ("manual" or "auto").
            file_name (str): Output file name for auto report.

        Returns:
            dict or str:
                - dict if mode="manual"
                - str message if mode="auto"

        Raises:
            ValueError: If mode is not "manual" or "auto".
        """
       if mode == "manual":
            return self.manual_report()
       
       elif mode == "auto":
            return self.auto_report()
       else :
            raise ValueError("mode must be either 'manual' or 'auto'")
