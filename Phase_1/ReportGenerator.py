import pandas as pd
from ydata_profiling import ProfileReport
from jinja2 import Template
import os


class  ReportGenerator:
    """
    A simple class for generating Exploratory Data Analysis (EDA) reports.

    The class supports two user-friendly modes:
    1. Basic Report:
        - Simple and structured summary
        - Includes key insights
        - Suitable for users and APIs

    2. Detailed Report:
        - Full automated report using ydata_profiling
        - Includes visualizations and deep analysis

    Attributes:
        data (pd.DataFrame): Input dataset
    """

    def __init__(self,data:pd.DataFrame):
        """
        Initialize the ReportGenerator with a dataset.

        Args:
            data (pd.DataFrame): The input dataset for analysis.
        """
        self.data =data
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
        Calculate missing values for each column and percentage.

        Returns:
            pd.DataFrame
        """
        missing = self.data.isnull().sum()
        percent = (missing/len(self.data)) *100
   
        return pd.DataFrame ({
             "Missing Count": missing,
            "Percentage (%)": percent
        })
    
    def insights(self):
        """
        Generate simple human-readable insights.

        Returns:
            list
        """
        insights = []
  
        # Missing values insight
        missing = self.data.isnull().sum()
        total_rows = len(self.data)

        for col,val in missing.items():
             percent = (val/total_rows )*100
             if percent > 0:
                  insights.append(f"Column '{col}' has {percent:.2f}% missing values")
        
        # Correlation (avoid duplicates)
        corr = self.data.corr(numeric_only=True)
        for i,col in enumerate(corr.columns):
             for j in range(i):
                  value =corr.iloc[i,j]
                  if abs(value) > 0.8 :
                       insights.append(
                        f"Strong correlation between '{col}' and '{corr.columns[j]}'"
                    )
             
 
        # Duplicates insight
        duplicates = self.data.duplicated().sum()
        if duplicates > 0 :
             insights.append(f"Dataset contains {duplicates} duplicate rows")
        

        return insights
   
    def overview(self):
        """
        Provide dataset overview.

        Returns:
            dict
        """
        return {
           "Shape":self.data.shape,
            "Columns": list(self.data.columns),
            "Data Types": self.data.dtypes.astype(str),
        }
   

    def manual_report(self):
        """
        Generate and render a complete Exploratory Data Analysis (EDA) HTML report.

        This method performs data analysis on the dataset and directly generates
        a ready-to-view HTML report using a Jinja2 template.

        Unlike a pure data-returning function, this method has side effects:
        it renders and saves an HTML file to disk.

        Workflow:
            1. Computes dataset statistics and quality metrics
            2. Generates structured insights from the data
            3. Loads an HTML Jinja2 template (report.html)
            4. Injects computed values into the template
            5. Renders a final HTML report
            6. Saves the output as 'final_report.html'

        Generated Report Includes:
            - Dataset shape (rows, columns)
            - Total missing values across dataset
            - Number of duplicate rows
            - Summary statistics table
            - Missing values table
            - Correlation matrix
            - Key insights (rule-based)
            - Column with highest missing percentage

        Side Effects:
            - Reads 'report.html' template file from disk
            - Writes output to 'final_report.html'

        Returns:
            str: Confirmation message indicating successful report generation
                and output file location.

        """
        
        report = {
            "shape": self.data.shape,
            "total_missing": self.data.isnull().sum().sum(),
            "duplicates": self.data.duplicated().sum(),
            "summary": self.summary_stats().to_html(),
            "missing": self.missing_values().to_html(),
            "correlation": self.correlation_matrix().to_html(),
            "insights": self.insights(),
            "top_missing": self.missing_values()["Percentage (%)"].idxmax(),
            "top_missing_val": self.missing_values()["Percentage (%)"].max()
        }

        
        template_path = os.path.join(self.BASE_DIR, "template.html")
        PROJECT_ROOT = os.path.abspath(os.path.join(self.BASE_DIR, os.pardir))
        output_path = os.path.join(PROJECT_ROOT, "final_report.html")         
        # check
        print("template exists:", os.path.exists(template_path))
        
        with open(template_path, "r", encoding="utf-8") as f:
            template = Template(f.read())

        html = template.render(**report)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return output_path


    def auto_report(self, file_name="report.html", explorative=True):
        """
        Generate a detailed automated EDA report using ydata_profiling.

        This report provides:
        - Full dataset analysis
        - Visualizations
        - Statistical summaries
        - Data quality checks

        Args:
            file_name (str): Output HTML file name for the report.
            explorative (bool): If True, enables deeper exploratory analysis.

        Returns:
            str: Confirmation message with saved file name.
        """
        profile = ProfileReport(self.data,explorative=explorative)
        PROJECT_ROOT = os.path.abspath(os.path.join(self.BASE_DIR, os.pardir))
        profile.to_file(os.path.join(PROJECT_ROOT, file_name))
        return f"Detailed report saved as {file_name}"
        
            
    def generate_report(self, mode="basic", file_name="report.html"):

        """
        Generate an EDA report based on user-selected mode.

        Modes:
            - "basic"    : returns structured analysis (summary, insights, statistics)
            - "detailed" : generates full automated HTML report using ydata_profiling

        Args:
            mode (str): Type of report to generate ("basic" or "detailed").
            file_name (str): Output file name for detailed report.

        Returns:
            dict or str:
                - dict: if mode="basic"
                - str: confirmation message if mode="detailed"

        Raises:
            ValueError: If invalid mode is provided.
        """

        if mode == "basic":
           self.manual_report()
           return "HTML report generated"

        elif mode == "detailed":
            self.auto_report(file_name=file_name)
            return "Detailed report generated"
    
        else:
            raise ValueError("mode must be 'basic' or 'detailed'")