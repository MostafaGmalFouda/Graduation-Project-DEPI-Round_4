import pandas as pd
from Phase_1.DataLoader import DataLoader
from Phase_1.DataValidator import DataValidator
from Phase_1.DataPreprocessor import DataPreprocessor
from Phase_1.OutlierHandler import OutlierHandler
from Phase_1.ReportGenerator import ReportGenerator


class EDAPipeline:
    """
    Orchestrates the full EDA pipeline by coordinating:
        DataLoader -> DataValidator -> DataPreprocessor -> OutlierHandler -> ReportGenerator

    Attributes:
        loader (DataLoader)             : Loads data from file.
        validator (DataValidator)       : Validates data quality.
        preprocessor (DataPreprocessor): Cleans and preprocesses data.
        outlier_handler (OutlierHandler): Detects and handles outliers.
        reporter (ReportGenerator)      : Generates EDA report.
    """

    def __init__(self, file_path: str):
        # ── Stage 1: Load ──────────────────────────────────────────────
        self.loader = DataLoader(file_path)
        self.loader.load()
        data = self.loader.get_data()

        # ── Stage 2: Validate ──────────────────────────────────────────
        self.validator = DataValidator(data)

        # ── Stage 3: Preprocess ────────────────────────────────────────
        self.preprocessor = DataPreprocessor(data)

        # ── Stage 4: Outlier Handler ───────────────────────────────────
        self.outlier_handler = OutlierHandler(data)

        # ── Stage 5: Report ────────────────────────────────────────────
        self.reporter = ReportGenerator(data)

    def run_pipeline(self) -> None:
        """
        Executes the full EDA pipeline end-to-end:

            1. Validate  — report nulls, types, duplicates on raw data
            2. Preprocess — handle nulls, convert types, remove duplicates
            3. Outliers  — detect (IQR + Z-Score) then cap outliers
            4. Report    — generate final EDA report on clean data
        """

        # ── Stage 1: Validate ──────────────────────────────────────────
        self._print_stage("1 — VALIDATING DATA")
        report = self.validator.report_issues()
        print(f"  • Nulls found    : {report['missing_values']}")
        print(f"  • Duplicates     : {report['duplicates']}")
        print(f"  • Column dtypes  : {report['data_types']}")

        # ── Stage 2: Preprocess ────────────────────────────────────────
        self._print_stage("2 — PREPROCESSING DATA")
        self.preprocessor.handle_nulls()
        self.preprocessor.convert_types()
        self.preprocessor.remove_duplicates()
        clean_data = self.preprocessor.get_clean_data()

        # ── Stage 3: Outlier Handling ──────────────────────────────────
        self._print_stage("3 — HANDLING OUTLIERS")
        self.outlier_handler = OutlierHandler(clean_data)

        print("\n  [IQR Detection]")
        iqr_result = self.outlier_handler.detect_iqr()
        for col, count in iqr_result.items():
            if count > 0:
                print(f"    • '{col}': {count} outlier(s)")

        print("\n  [Z-Score Detection]")
        zscore_result = self.outlier_handler.detect_zscore()
        for col, count in zscore_result.items():
            if count > 0:
                print(f"    • '{col}': {count} outlier(s)")

        clean_data = self.outlier_handler.cap_outliers()

        # ── Stage 4: Report ────────────────────────────────────────────
        self._print_stage("4 — GENERATING REPORT")
        self.reporter = ReportGenerator(clean_data)
        self.reporter.generate_report()

        self._print_stage("PIPELINE COMPLETE ✓")
        print(f"  Final dataset shape: {clean_data.shape}")

    @staticmethod
    def _print_stage(title: str) -> None:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print(f"{'=' * 50}")