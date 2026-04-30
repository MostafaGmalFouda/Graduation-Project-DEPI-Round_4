import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — safe for saving without display
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Global Seaborn Style ───────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")


class DataVisualizer:
    """
    A class to generate rich visualizations for Exploratory Data Analysis (EDA).

    Supports:
        - Bivariate & Multivariate (General): Summary dashboard, missing values matrix,
          correlation heatmap.
        - Num vs Num (Numerical): 2D scatter, 3D scatter, joint plot.
        - Cat vs Cat (Categorical): Stacked bar, cross-tabulation heatmap,
          violin by category, facet grid, bubble chart.

    Attributes:
        data     (pd.DataFrame) : The clean dataset to visualize.
        schema   (dict)         : Maps column names → 'num' or 'cat'.
                                  Auto-detected if not provided; can be overridden.
        plots_dir (str)         : Directory where PNG/HTML plots are saved.

    Methods:
        -- Bivariate & Multivariate - General --
        generate_summary_dashboard()                                  : Overview dashboard.
        plot_missing_values_matrix()                                  : Missing values heatmap.
        plot_correlation_heatmap()                                    : Correlation heatmap.

        -- Num vs Num (Numerical) --
        plot_scatter_2d(col1, col2, color_col, save)                 : 2D scatter plot.
        plot_scatter_3d(col1, col2, col3, color_col, save)           : 3D scatter plot.
        plot_joint_plot(col1, col2, kind, save)                      : Joint distribution plot.

        -- Cat vs Cat (Categorical) --
        plot_stacked_bar(col1, col2, normalize, save)                : Stacked bar chart.
        plot_cross_tabulation(col1, col2, save)                      : Cross-tab heatmap.
        plot_violin_plot_by_category(num_col, cat_col, save)         : Violin plot.
        plot_facet_grid(num_cols, cat_col, save)                     : Facet grid of histograms.
        plot_bubble_chart(x, y, size, color, save)                   : Bubble chart.
    """

    # ── Type tags used in schema ──────────────────────────────────────────────
    _NUM = "num"
    _CAT = "cat"

    def __init__(self, data: pd.DataFrame, schema: dict = None):
        """
        Initialize DataVisualizer.

        Args:
            data   (pd.DataFrame): Clean dataset (output of DataPreprocessor or OutlierHandler).
            schema (dict)        : Optional. Maps column names to 'num' or 'cat'.
                                   Auto-detected from dtypes if not provided.
                                   Partial schema allowed — missing columns are auto-detected.
        """
        self.data = data.copy()
        self.schema = self._build_schema(schema)

        # ── Output directory ─────────────────────────────────────────────────
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.plots_dir = os.path.join(BASE_DIR, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

        print(f"DataVisualizer initialized.")
        print(f"  • Rows    : {self.data.shape[0]}")
        print(f"  • Columns : {self.data.shape[1]}")
        print(f"  • Schema  : {self.schema}")
        print(f"  • Plots will be saved to: {self.plots_dir}\n")

    # ══════════════════════════════════════════════════════════════════════════
    #  PRIVATE HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _build_schema(self, user_schema: dict) -> dict:
        """
        Build a full schema by:
          1. Auto-detecting from dtypes.
          2. Overriding with any values provided in user_schema.

        Args:
            user_schema (dict): User-supplied partial or full schema.

        Returns:
            dict: Complete schema for all columns.
        """
        auto_schema = {}
        for col in self.data.columns:
            if self.data[col].dtype in ['int8', 'int16', 'int32', 'int64',
                                         'float16', 'float32', 'float64']:
                auto_schema[col] = self._NUM
            else:
                auto_schema[col] = self._CAT

        if user_schema:
            for col, dtype in user_schema.items():
                if col in auto_schema:
                    auto_schema[col] = dtype
                    print(f"  Schema override: '{col}' → '{dtype}'")

        return auto_schema

    def _get_cols_by_type(self, col_type: str) -> list:
        """Return list of column names matching the given schema type."""
        return [col for col, t in self.schema.items() if t == col_type]

    def _save_matplotlib(self, fig: plt.Figure, filename: str) -> str:
        """Save a Matplotlib/Seaborn figure as PNG and show it."""
        path = os.path.join(self.plots_dir, filename)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.show()
        plt.close(fig)
        print(f"  ✔ Saved: {path}")
        return path

    def _save_plotly(self, fig, filename: str) -> str:
        """Save a Plotly figure as interactive HTML and show it."""
        path = os.path.join(self.plots_dir, filename)
        fig.write_html(path)
        fig.show()
        print(f"  ✔ Saved: {path}")
        return path

    def _validate_cols(self, cols: list) -> None:
        """Raise ValueError if any column is not in the dataset."""
        for col in cols:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in dataset.")

    # ══════════════════════════════════════════════════════════════════════════
    #  BIVARIATE & MULTIVARIATE — GENERAL
    # ══════════════════════════════════════════════════════════════════════════

    def generate_summary_dashboard(self, save: bool = True) -> str:
        """
        Generate a high-level summary dashboard showing:
          - Distribution of all numeric columns (histograms + KDE)
          - Value counts of all categorical columns (bar charts)

        Args:
            save (bool): Save the plot to file. Default True.

        Returns:
            str: Path to saved PNG file.
        """
        try:
            num_cols = self._get_cols_by_type(self._NUM)
            cat_cols = self._get_cols_by_type(self._CAT)
            all_cols = num_cols + cat_cols

            if not all_cols:
                raise ValueError("No columns available for summary dashboard.")

            n = len(all_cols)
            ncols = 3
            nrows = (n + ncols - 1) // ncols

            fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 4))
            axes = axes.flatten()
            fig.suptitle("EDA Summary Dashboard", fontsize=18, fontweight="bold", y=1.01)

            for i, col in enumerate(all_cols):
                ax = axes[i]
                if col in num_cols:
                    sns.histplot(self.data[col].dropna(), kde=True, ax=ax, color="steelblue")
                    ax.set_title(f"[Numeric] {col}", fontsize=11)
                else:
                    top_vals = self.data[col].value_counts().head(10)
                    sns.barplot(x=top_vals.values, y=top_vals.index.astype(str), ax=ax, palette="muted")
                    ax.set_title(f"[Categorical] {col}", fontsize=11)
                ax.set_xlabel("")

            # Hide unused axes
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            print("Generating Summary Dashboard...")
            return self._save_matplotlib(fig, "summary_dashboard.png") if save else None

        except Exception as e:
            raise Exception(f"Error in generate_summary_dashboard: {e}")

    def plot_missing_values_matrix(self, save: bool = True) -> str:
        """
        Visualize missing values across all columns using a heatmap.
        Columns with no missing values are excluded from the plot.

        Args:
            save (bool): Save the plot to file. Default True.

        Returns:
            str: Path to saved PNG file.
        """
        try:
            missing_cols = [col for col in self.data.columns if self.data[col].isnull().any()]

            if not missing_cols:
                print("  No missing values found — skipping missing values matrix.")
                return None

            fig, ax = plt.subplots(figsize=(14, 6))
            missing_matrix = self.data[missing_cols].isnull().astype(int)

            sns.heatmap(
                missing_matrix.T,
                ax=ax,
                cmap="YlOrRd",
                cbar_kws={"label": "Missing (1) / Present (0)"},
                linewidths=0.3,
                yticklabels=missing_cols
            )
            ax.set_title("Missing Values Matrix", fontsize=15, fontweight="bold")
            ax.set_xlabel("Row Index")
            ax.set_ylabel("Columns")

            plt.tight_layout()
            print("Generating Missing Values Matrix...")
            return self._save_matplotlib(fig, "missing_values_matrix.png") if save else None

        except Exception as e:
            raise Exception(f"Error in plot_missing_values_matrix: {e}")

    def plot_correlation_heatmap(self, save: bool = True) -> str:
        """
        Generate a Seaborn correlation heatmap for all numeric columns.
        Annotated with correlation coefficients.

        Args:
            save (bool): Save the plot to file. Default True.

        Returns:
            str: Path to saved PNG file.
        """
        try:
            num_cols = self._get_cols_by_type(self._NUM)

            if len(num_cols) < 2:
                raise ValueError("Need at least 2 numeric columns for a correlation heatmap.")

            corr = self.data[num_cols].corr()

            fig, ax = plt.subplots(figsize=(max(8, len(num_cols)), max(6, len(num_cols) - 1)))
            mask = corr.isnull()  # Mask NaN values

            sns.heatmap(
                corr,
                ax=ax,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                mask=mask,
                linewidths=0.5,
                square=True,
                cbar_kws={"shrink": 0.8}
            )
            ax.set_title("Correlation Heatmap", fontsize=15, fontweight="bold")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            print("Generating Correlation Heatmap...")
            return self._save_matplotlib(fig, "correlation_heatmap.png") if save else None

        except Exception as e:
            raise Exception(f"Error in plot_correlation_heatmap: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    #  NUM vs NUM — NUMERICAL
    # ══════════════════════════════════════════════════════════════════════════

    def plot_scatter_2d(
        self,
        col1: str,
        col2: str,
        color_col: str = None,
        save: bool = True
    ) -> str:
        """
        Generate an interactive 2D scatter plot (Plotly) between two numeric columns.
        Optionally color points by a third column (numeric or categorical).

        Args:
            col1      (str) : X-axis column name (numeric).
            col2      (str) : Y-axis column name (numeric).
            color_col (str) : Optional column to color points by.
            save     (bool) : Save the plot as HTML. Default True.

        Returns:
            str: Path to saved HTML file.
        """
        try:
            self._validate_cols([col1, col2] + ([color_col] if color_col else []))

            fig = px.scatter(
                self.data,
                x=col1,
                y=col2,
                color=color_col,
                title=f"2D Scatter: {col1} vs {col2}" + (f" (colored by {color_col})" if color_col else ""),
                opacity=0.7,
                template="plotly_white",
                hover_data=self.data.columns.tolist()
            )
            fig.update_traces(marker=dict(size=6))
            fig.update_layout(title_font_size=16)

            print(f"Generating 2D Scatter: {col1} vs {col2}...")
            return self._save_plotly(fig, f"scatter_2d_{col1}_vs_{col2}.html") if save else None

        except Exception as e:
            raise Exception(f"Error in plot_scatter_2d: {e}")

    def plot_scatter_3d(
        self,
        col1: str,
        col2: str,
        col3: str,
        color_col: str = None,
        save: bool = True
    ) -> str:
        """
        Generate an interactive 3D scatter plot (Plotly) across three numeric columns.
        Optionally color points by a fourth column.

        Args:
            col1      (str) : X-axis column name (numeric).
            col2      (str) : Y-axis column name (numeric).
            col3      (str) : Z-axis column name (numeric).
            color_col (str) : Optional column to color points by.
            save     (bool) : Save the plot as HTML. Default True.

        Returns:
            str: Path to saved HTML file.
        """
        try:
            self._validate_cols([col1, col2, col3] + ([color_col] if color_col else []))

            fig = px.scatter_3d(
                self.data,
                x=col1,
                y=col2,
                z=col3,
                color=color_col,
                title=f"3D Scatter: {col1} / {col2} / {col3}",
                opacity=0.75,
                template="plotly_white"
            )
            fig.update_traces(marker=dict(size=4))
            fig.update_layout(title_font_size=16)

            print(f"Generating 3D Scatter: {col1} / {col2} / {col3}...")
            return self._save_plotly(fig, f"scatter_3d_{col1}_{col2}_{col3}.html") if save else None

        except Exception as e:
            raise Exception(f"Error in plot_scatter_3d: {e}")

    def plot_joint_plot(
        self,
        col1: str,
        col2: str,
        kind: str = "scatter",
        save: bool = True
    ) -> str:
        """
        Generate a joint distribution plot (Seaborn) showing:
        - Central plot : scatter / kde / hex / reg between col1 and col2.
        - Marginal plots: distribution of each variable independently.

        Args:
            col1 (str) : X-axis column name (numeric).
            col2 (str) : Y-axis column name (numeric).
            kind (str) : Plot kind — 'scatter', 'kde', 'hex', 'reg'. Default 'scatter'.
            save (bool): Save the plot to file. Default True.

        Returns:
            str: Path to saved PNG file.
        """
        try:
            self._validate_cols([col1, col2])
            valid_kinds = ("scatter", "kde", "hex", "reg")
            if kind not in valid_kinds:
                raise ValueError(f"kind must be one of {valid_kinds}.")

            clean = self.data[[col1, col2]].dropna()
            g = sns.jointplot(data=clean, x=col1, y=col2, kind=kind, height=8, palette="muted")
            g.fig.suptitle(f"Joint Plot: {col1} vs {col2} ({kind})", y=1.02, fontsize=14, fontweight="bold")

            print(f"Generating Joint Plot: {col1} vs {col2} ({kind})...")
            return self._save_matplotlib(g.fig, f"joint_plot_{col1}_vs_{col2}.png") if save else None

        except Exception as e:
            raise Exception(f"Error in plot_joint_plot: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    #  CAT vs CAT — CATEGORICAL
    # ══════════════════════════════════════════════════════════════════════════

    def plot_stacked_bar(
        self,
        col1: str,
        col2: str,
        normalize: bool = False,
        save: bool = True
    ) -> str:
        """
        Generate a stacked bar chart showing the distribution of col2 within each
        category of col1.

        Args:
            col1      (str)  : Primary grouping column (categorical).
            col2      (str)  : Secondary grouping column (categorical).
            normalize (bool) : If True, plot proportions (0–100%) instead of counts.
            save      (bool) : Save the plot to file. Default True.

        Returns:
            str: Path to saved PNG file.
        """
        try:
            self._validate_cols([col1, col2])

            ct = pd.crosstab(self.data[col1], self.data[col2], normalize="index" if normalize else False)
            ylabel = "Proportion (%)" if normalize else "Count"

            fig, ax = plt.subplots(figsize=(12, 6))
            ct.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", edgecolor="white", linewidth=0.5)
            ax.set_title(f"Stacked Bar: {col1} by {col2}", fontsize=14, fontweight="bold")
            ax.set_xlabel(col1)
            ax.set_ylabel(ylabel)
            ax.legend(title=col2, bbox_to_anchor=(1.01, 1), loc="upper left")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            fname = f"stacked_bar_{col1}_by_{col2}.png"
            print(f"Generating Stacked Bar: {col1} by {col2}...")
            return self._save_matplotlib(fig, fname) if save else None

        except Exception as e:
            raise Exception(f"Error in plot_stacked_bar: {e}")

    def plot_cross_tabulation(
        self,
        col1: str,
        col2: str,
        save: bool = True
    ) -> str:
        """
        Generate a cross-tabulation heatmap (Seaborn) showing the frequency of
        each combination of (col1, col2).

        Args:
            col1 (str) : Row variable (categorical).
            col2 (str) : Column variable (categorical).
            save (bool): Save the plot to file. Default True.

        Returns:
            str: Path to saved PNG file.
        """
        try:
            self._validate_cols([col1, col2])

            ct = pd.crosstab(self.data[col1], self.data[col2])

            fig, ax = plt.subplots(figsize=(max(8, len(ct.columns)), max(6, len(ct))))
            sns.heatmap(
                ct,
                ax=ax,
                annot=True,
                fmt="d",
                cmap="Blues",
                linewidths=0.4,
                cbar_kws={"label": "Count"}
            )
            ax.set_title(f"Cross-Tabulation: {col1} vs {col2}", fontsize=14, fontweight="bold")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            fname = f"cross_tab_{col1}_vs_{col2}.png"
            print(f"Generating Cross-Tabulation: {col1} vs {col2}...")
            return self._save_matplotlib(fig, fname) if save else None

        except Exception as e:
            raise Exception(f"Error in plot_cross_tabulation: {e}")

    def plot_violin_plot_by_category(
        self,
        num_col: str,
        cat_col: str,
        save: bool = True
    ) -> str:
        """
        Generate a violin plot showing the distribution of a numeric column
        split by the categories of a categorical column.

        Args:
            num_col (str) : Numeric column to plot on Y-axis.
            cat_col (str) : Categorical column to group by on X-axis.
            save    (bool): Save the plot to file. Default True.

        Returns:
            str: Path to saved PNG file.
        """
        try:
            self._validate_cols([num_col, cat_col])

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.violinplot(
                data=self.data,
                x=cat_col,
                y=num_col,
                ax=ax,
                palette="muted",
                inner="quartile"
            )
            ax.set_title(f"Violin Plot: {num_col} by {cat_col}", fontsize=14, fontweight="bold")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            fname = f"violin_{num_col}_by_{cat_col}.png"
            print(f"Generating Violin Plot: {num_col} by {cat_col}...")
            return self._save_matplotlib(fig, fname) if save else None

        except Exception as e:
            raise Exception(f"Error in plot_violin_plot_by_category: {e}")

    def plot_facet_grid(
        self,
        num_cols: list,
        cat_col: str,
        save: bool = True
    ) -> str:
        """
        Generate a Seaborn FacetGrid showing histograms (+ KDE) of multiple numeric
        columns, each faceted by the unique categories of cat_col.

        Args:
            num_cols (list) : List of numeric column names to plot.
            cat_col  (str)  : Categorical column used to create one facet per category.
            save     (bool) : Save the plot to file. Default True.

        Returns:
            str: Path to saved PNG file.
        """
        try:
            self._validate_cols(num_cols + [cat_col])

            # Melt numeric columns into long format for FacetGrid
            melted = self.data[num_cols + [cat_col]].melt(id_vars=cat_col, var_name="Variable", value_name="Value")

            g = sns.FacetGrid(melted, col="Variable", row=cat_col, height=3.5, aspect=1.4, sharey=False)
            g.map_dataframe(sns.histplot, x="Value", kde=True, color="steelblue")
            g.set_titles(col_template="{col_name}", row_template="{row_name}")
            g.set_axis_labels("Value", "Count")
            g.fig.suptitle(f"Facet Grid: {num_cols} by {cat_col}", fontsize=13, fontweight="bold", y=1.01)

            fname = f"facet_grid_by_{cat_col}.png"
            print(f"Generating Facet Grid for {num_cols} by {cat_col}...")
            return self._save_matplotlib(g.fig, fname) if save else None

        except Exception as e:
            raise Exception(f"Error in plot_facet_grid: {e}")

    def plot_bubble_chart(
        self,
        x: str,
        y: str,
        size: str,
        color: str = None,
        save: bool = True
    ) -> str:
        """
        Generate an interactive bubble chart (Plotly).
        Handles any combination of numeric and categorical columns:
          - x     : numeric or categorical  (X-axis)
          - y     : numeric                 (Y-axis)
          - size  : numeric                 (bubble size)
          - color : numeric or categorical  (bubble color, optional)

        Args:
            x     (str) : X-axis column.
            y     (str) : Y-axis column (numeric).
            size  (str) : Column to map to bubble size (numeric).
            color (str) : Optional column to color bubbles by.
            save  (bool): Save the plot as HTML. Default True.

        Returns:
            str: Path to saved HTML file.
        """
        try:
            cols = [x, y, size] + ([color] if color else [])
            self._validate_cols(cols)

            # Ensure size column has no negatives (Plotly requirement)
            plot_data = self.data.copy()
            min_size = plot_data[size].min()
            if min_size < 0:
                plot_data[size] = plot_data[size] - min_size  # Shift to zero-based

            fig = px.scatter(
                plot_data,
                x=x,
                y=y,
                size=size,
                color=color,
                title=f"Bubble Chart: {x} vs {y} (size={size}" + (f", color={color}" if color else "") + ")",
                template="plotly_white",
                opacity=0.75,
                size_max=60,
                hover_data=self.data.columns.tolist()
            )
            fig.update_layout(title_font_size=15)

            fname = f"bubble_chart_{x}_vs_{y}.html"
            print(f"Generating Bubble Chart: {x} vs {y} (size={size})...")
            return self._save_plotly(fig, fname) if save else None

        except Exception as e:
            raise Exception(f"Error in plot_bubble_chart: {e}")
