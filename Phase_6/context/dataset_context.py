import pandas as pd


class DatasetContext:
    """
    Keeps a snapshot of the dataset at two stages:
      - raw   : right after upload, before any cleaning
      - clean : after the preprocessing / cleaning pipeline runs

    Both snapshots stay available at the same time so the chatbot can
    answer questions about the data before AND after cleaning.
    """

    def __init__(self):
        self.raw = {}
        self.clean = {}

    def _summarize(self, df: pd.DataFrame) -> dict:
        return {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": int(df.duplicated().sum()),
            "describe": df.describe(include="all").fillna("").to_dict(),
        }

    def update_raw(self, df: pd.DataFrame):
        self.raw = self._summarize(df)

    def update_clean(self, df: pd.DataFrame):
        self.clean = self._summarize(df)

    # kept for backward compatibility with any old call sites
    def update(self, df: pd.DataFrame):
        self.update_raw(df)

    def get_context(self):
        return {
            "raw": self.raw,
            "clean": self.clean,
        }