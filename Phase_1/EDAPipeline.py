# import pandas as pd
# from Phase_1.DataLoader import DataLoader
# from Phase_1.DataValidator import DataValidator
# from Phase_1.DataPreprocessor import DataPreprocessor
# from Phase_1.OutlierHandler import OutlierHandler
# from Phase_1.ReportGenerator import ReportGenerator


# class EDAPipeline:
#     """
#     Orchestrates the full EDA pipeline by coordinating:
#         DataLoader -> DataValidator -> DataPreprocessor -> OutlierHandler -> ReportGenerator

#     Attributes:
#         loader (DataLoader)             : Loads data from file.
#         validator (DataValidator)       : Validates data quality.
#         preprocessor (DataPreprocessor): Cleans and preprocesses data.
#         outlier_handler (OutlierHandler): Detects and handles outliers.
#         reporter (ReportGenerator)      : Generates EDA report.
#     """

#     def __init__(self, file_path: str):
#         # ── Stage 1: Load ──────────────────────────────────────────────
#         self.loader = DataLoader(file_path)
#         self.loader.load()
#         data = self.loader.get_data()

#         # ── Stage 2: Validate ──────────────────────────────────────────
#         self.validator = DataValidator(data)

#         # ── Stage 3: Preprocess ────────────────────────────────────────
#         self.preprocessor = DataPreprocessor(data)

#         # ── Stage 4: Outlier Handler ───────────────────────────────────
#         self.outlier_handler = OutlierHandler(data)

#         # ── Stage 5: Report ────────────────────────────────────────────
#         self.reporter = ReportGenerator(data)

#     def run_pipeline(self) -> None:
#         """
#         Executes the full EDA pipeline end-to-end:

#             1. Validate  — report nulls, types, duplicates on raw data
#             2. Preprocess — handle nulls, convert types, remove duplicates
#             3. Outliers  — detect (IQR + Z-Score) then cap outliers
#             4. Report    — generate final EDA report on clean data
#         """

#         # ── Stage 1: Validate ──────────────────────────────────────────
#         self._print_stage("1 — VALIDATING DATA")
#         report = self.validator.report_issues()
#         print(f"  • Nulls found    : {report['missing_values']}")
#         print(f"  • Duplicates     : {report['duplicates']}")
#         print(f"  • Column dtypes  : {report['data_types']}")

#         # ── Stage 2: Preprocess ────────────────────────────────────────
#         self._print_stage("2 — PREPROCESSING DATA")
#         self.preprocessor.handle_nulls()
#         self.preprocessor.convert_types()
#         self.preprocessor.remove_duplicates()
#         clean_data = self.preprocessor.get_clean_data()

#         # ── Stage 3: Outlier Handling ──────────────────────────────────
#         self._print_stage("3 — HANDLING OUTLIERS")
#         self.outlier_handler = OutlierHandler(clean_data)

#         print("\n  [IQR Detection]")
#         iqr_result = self.outlier_handler.detect_iqr()
#         for col, count in iqr_result.items():
#             if count > 0:
#                 print(f"    • '{col}': {count} outlier(s)")

#         print("\n  [Z-Score Detection]")
#         zscore_result = self.outlier_handler.detect_zscore()
#         for col, count in zscore_result.items():
#             if count > 0:
#                 print(f"    • '{col}': {count} outlier(s)")

#         clean_data = self.outlier_handler.cap_outliers()

#         # ── Stage 4: Report ────────────────────────────────────────────
#         self._print_stage("4 — GENERATING REPORT")
#         self.reporter = ReportGenerator(clean_data)
#         self.reporter.generate_report()

#         self._print_stage("PIPELINE COMPLETE ✓")
#         print(f"  Final dataset shape: {clean_data.shape}")

#     @staticmethod
#     def _print_stage(title: str) -> None:
#         print(f"\n{'=' * 50}")
#         print(f"  {title}")
#         print(f"{'=' * 50}")
import pandas as pd
from Phase_1.DataLoader import DataLoader
from Phase_1.DataValidator import DataValidator
from Phase_1.DataPreprocessor import DataPreprocessor
from Phase_1.OutlierHandler import OutlierHandler
from Phase_1.ReportGenerator import ReportGenerator


class EDAPipeline:
    """
    Orchestrates the full EDA pipeline:

        Load -> Data Understanding -> General EDA -> Cleaning
              -> (detect text columns) -> Text EDA (if any) -> Report

    General EDA always runs first (nulls, dtypes, duplicates, outliers,
    correlation). Only AFTER that does the pipeline check whether any
    text/NLP columns exist and run a separate Text EDA stage for them.

    Attributes:
        loader (DataLoader)              : Loads data from file.
        validator (DataValidator)        : Validates data quality.
        preprocessor (DataPreprocessor)  : Cleans and preprocesses data.
        outlier_handler (OutlierHandler) : Detects and handles outliers.
        reporter (ReportGenerator)       : Generates EDA report.
        text_columns (list[str])         : Detected text/NLP columns.
    """

    # Columns with average word count above this are treated as "text"
    # rather than plain categorical (e.g. "Yes"/"No", city names, etc.)
    TEXT_AVG_WORD_THRESHOLD = 3

    def __init__(self, file_path: str, target_column: str = None):
        self.target_column = target_column

        # ── Stage 0: Load ──────────────────────────────────────────────
        self.loader = DataLoader(file_path)
        self.loader.load()
        data = self.loader.get_data()

        # Objects are created lazily inside run_pipeline() once we have
        # the real (possibly cleaned) data — avoids building throwaway
        # instances here that just get overwritten later.
        self.validator = None
        self.preprocessor = None
        self.outlier_handler = None
        self.reporter = None

        self._raw_data = data
        self.text_columns = []
        self.clean_data = None

    def run_pipeline(self) -> pd.DataFrame:
        """
        Executes the full pipeline end-to-end and returns the final
        cleaned DataFrame.

            0. Data Understanding — shape, dtypes, target check
            1. General EDA        — nulls, duplicates, dtypes, outliers
            2. Cleaning           — handle nulls, convert types, dedupe
            3. Detect text cols   — decide if NLP branch is needed
            4. Text EDA           — only runs if text columns exist
            5. Report             — final EDA report on clean data
        """

        # ── Stage 0: Data Understanding ──────────────────────────────
        self._print_stage("0 — DATA UNDERSTANDING")
        data = self._raw_data
        print(f"  • Shape           : {data.shape[0]} rows, {data.shape[1]} columns")
        print(f"  • Columns         : {list(data.columns)}")
        print(f"  • Dtypes          : {data.dtypes.to_dict()}")
        if self.target_column:
            has_target = self.target_column in data.columns
            print(f"  • Target column   : '{self.target_column}' "
                  f"({'found' if has_target else 'NOT FOUND'})")
        else:
            print("  • Target column   : not specified")

        # ── Stage 1: General EDA (validation, on RAW data) ───────────
        self._print_stage("1 — GENERAL EDA (raw data)")
        self.validator = DataValidator(data)
        report = self.validator.report_issues()
        print(f"  • Nulls found     : {report['missing_values']}")
        print(f"  • Duplicates      : {report['duplicates']}")
        print(f"  • Column dtypes   : {report['data_types']}")

        numeric_cols = data.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) > 1:
            print("\n  [Correlation matrix — numeric columns]")
            print(data[numeric_cols].corr())

        if self.target_column and self.target_column in data.columns:
            print(f"\n  [Class distribution — '{self.target_column}']")
            print(data[self.target_column].value_counts())

        # ── Stage 2: Cleaning ─────────────────────────────────────────
        self._print_stage("2 — CLEANING DATA")
        # Pass a copy so DataPreprocessor never mutates self._raw_data
        # (or the DataFrame self.validator is still holding a reference to)
        self.preprocessor = DataPreprocessor(data.copy())
        self.preprocessor.handle_nulls()
        self.preprocessor.convert_types()
        self.preprocessor.remove_duplicates()
        clean_data = self.preprocessor.get_clean_data()

        # ── Stage 2b: Outlier handling (numeric columns only) ────────
        self._print_stage("2b — HANDLING OUTLIERS (numeric)")
        self.outlier_handler = OutlierHandler(clean_data)

        print("\n  [IQR Detection]")
        for col, count in self.outlier_handler.detect_iqr().items():
            if count > 0:
                print(f"    • '{col}': {count} outlier(s)")

        print("\n  [Z-Score Detection]")
        for col, count in self.outlier_handler.detect_zscore().items():
            if count > 0:
                print(f"    • '{col}': {count} outlier(s)")

        clean_data = self.outlier_handler.cap_outliers()

       # ── Stage 3: Detect text/NLP columns (via ReportGenerator) ─────
        self._print_stage("3 — DETECTING TEXT COLUMNS")
        self.reporter = ReportGenerator(clean_data, target=self.target_column)
        self.text_columns = self.reporter.feature_types()["text"]
        if self.text_columns:
            print(f"  • Text columns found: {self.text_columns}")
        else:
            print("  • No text columns found — this dataset is structured-only.")

        # ── Stage 4: Text EDA (only if text columns exist) ────────────
        # ── Stage 4: Text EDA (strict mode) ──────────────────────────────
        if self.text_columns:
    # Optional: Only trigger if text columns make up > 30% of dataset
             text_ratio = len(self.text_columns) / len(clean_data.columns)
             if text_ratio >= 0.3:  # e.g., at least 30% of columns are text
                self._print_stage("4 — TEXT EDA")
                print(self.reporter.text_summary())
                print(self.reporter.nlp_readiness())
             else:
               print(f"[Text EDA] Found {len(self.text_columns)} text column(s) "
                f"({text_ratio:.0%} of data), but threshold is 30%. Skipping deep NLP.")
        else:
         self._print_stage("4 — TEXT EDA (skipped, no text columns)")

        # ── Stage 5: Final Report ──────────────────────────────────────
        self._print_stage("5 — GENERATING REPORT")
    
        self.reporter.generate_report()

        self._print_stage("PIPELINE COMPLETE ✓")
        print(f"  Final dataset shape: {clean_data.shape}")
        if self.text_columns:
            print(f"  Recommended next step: NLP preprocessing on "
                  f"{self.text_columns}, then merge with numeric features "
                  f"before Machine Learning.")
        else:
            print("  Recommended next step: proceed directly to Machine Learning.")

        self.clean_data = clean_data
        return clean_data

    # # def _detect_text_columns(self, data: pd.DataFrame) -> list:
    # #     """
    # #     Distinguishes free-text (NLP) columns from short categorical
    # #     object columns. A column is treated as "text" if it's non-numeric
    # #     AND its average word count exceeds TEXT_AVG_WORD_THRESHOLD.

    # #     e.g. "Male"/"Female" -> categorical (1 word avg) -> NOT text
    # #          "Very good service, will buy again" -> text (5+ words avg)
    # #     """
    # #     candidates = data.select_dtypes(include=["object", "string"]).columns
    # #     text_cols = []
    # #     for col in candidates:
    # #         non_null = data[col].dropna().astype(str)
    # #         if non_null.empty:
    # #             continue
    # #         avg_words = non_null.str.split().str.len().mean()
    # #         if avg_words > self.TEXT_AVG_WORD_THRESHOLD:
    # #             text_cols.append(col)
    # #     return text_cols

    # @staticmethod
    # def _run_text_eda(data: pd.DataFrame, col: str) -> None:
    #     """Basic Text EDA for a single detected text column."""
    #     series = data[col].dropna().astype(str)
    #     empty_ratio = (data[col].isna().sum() +
    #                    (data[col].astype(str).str.strip() == "").sum()) / len(data)

    #     lengths = series.str.len()
    #     word_counts = series.str.split().str.len()

    #     print(f"\n  [Column: '{col}']")
    #     print(f"    • Empty/missing ratio : {empty_ratio:.2%}")
    #     print(f"    • Avg char length     : {lengths.mean():.1f}")
    #     print(f"    • Avg word count      : {word_counts.mean():.1f}")
    #     print(f"    • Max word count      : {word_counts.max()}")

    #     # Most frequent words (simple whitespace split, lowercase)
    #     all_words = (
    #         series.str.lower()
    #         .str.replace(r"[^\w\s]", "", regex=True)
    #         .str.split()
    #         .explode()
    #     )
    #     top_words = all_words.value_counts().head(10)
    #     print(f"    • Top 10 words        :")
    #     for word, count in top_words.items():
    #         print(f"        - {word}: {count}")

    #     # Top bigrams
    #     def bigrams(tokens):
    #         return [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]

    #     all_bigrams = series.str.lower().str.split().apply(bigrams).explode()
    #     top_bigrams = all_bigrams.value_counts().head(5)
    #     print(f"    • Top 5 bigrams       :")
    #     for bg, count in top_bigrams.items():
    #         print(f"        - {bg}: {count}")

    @staticmethod
    def _print_stage(title: str) -> None:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print(f"{'=' * 50}")