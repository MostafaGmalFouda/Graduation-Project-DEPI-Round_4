import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "json"  # Prevent any browser/tab from opening


# --- AI Cyberpunk Theme Colors ---
BG_COLOR = "#0b0b18"
GRID_COLOR = "#1f1f3a"
color_3= "#9cfbea7c"
color_4= "#77f8e185"
color_5= "#42efcfa5"
color_6= "#e29cfb7b"
color_7= "#c483f6b7"
color_8= "#d359fb7b"
TEXT_COLOR = "#e0e0e0"
ACCENT_PURPLE = "#a855f7"
ACCENT_CYAN = "#06d49d"
PALETTE = [ACCENT_PURPLE, ACCENT_CYAN, "#a248eccf", "#8959f9c8", "#0ee98ed8"]
COLORY=[color_3, color_4,color_5,ACCENT_CYAN,color_6,color_7,color_8,ACCENT_PURPLE]
# --- Global Seaborn Style Configuration ---
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": BG_COLOR,
    "figure.facecolor": BG_COLOR,
    "grid.color": GRID_COLOR,
    "axes.edgecolor": GRID_COLOR,
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "axes.titlecolor": ACCENT_CYAN
})
custom_cmap = sns.blend_palette([color_3, color_4,color_5,ACCENT_CYAN,color_6,color_7,color_8,ACCENT_PURPLE], as_cmap=True)

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
    
    def _low_cardinality_cat_cols(self, cat_cols: list, max_unique: int = 30) -> list:
        """
        Filter categorical columns down to ones actually safe for PAIRWISE
        plots (stacked bar, cross-tab heatmap, violin-by-category, facet
        grid, bubble color) -- i.e. real categories, not free text or
        high-cardinality identifiers.

        Without this, a free-text column (e.g. a preserved NLP review
        column with hundreds/thousands of unique values) reaching
        plot_cross_tabulation()/plot_stacked_bar() builds a crosstab sized
        by category count with no cap -- e.g.
        figsize=(max(8, len(ct.columns)), max(6, len(ct))) -- so 500 unique
        text values means a ~500-inch-tall heatmap that takes forever (or
        effectively hangs) to render and save. Single-column overviews
        (_plot_single_categorical) stay safe on their own since they
        already cap to value_counts().head(10), so this filter only needs
        to apply to the pairwise/multi-column plots.
        """
        safe = []
        n = len(self.data)
        for col in cat_cols:
            try:
                nunique = self.data[col].nunique(dropna=True)
            except TypeError:
                continue  # unhashable column contents -- skip entirely
            if nunique <= max_unique and (n == 0 or nunique / n < 0.5):
                safe.append(col)
        return safe
    
    def _apply_dark_layout_plotly(self, fig):
        fig.update_layout(
            paper_bgcolor=BG_COLOR,
            plot_bgcolor=BG_COLOR,
            font_color=TEXT_COLOR,
            margin=dict(t=50, l=25, r=25, b=25),
            xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
        )
        return fig

    def _save_matplotlib(self, fig, filename: str) -> str:
        """Save a Matplotlib/Seaborn figure as PNG (no display)."""
        path = os.path.join(self.plots_dir, filename)
        fig.savefig(path, bbox_inches="tight", dpi=150, facecolor=BG_COLOR)
        plt.close(fig)
        print(f"  ✔ Saved: {path}")
        return path

    def _save_plotly(self, fig, filename: str) -> str:
        """Save a Plotly figure as interactive HTML (no display)."""
        path = os.path.join(self.plots_dir, filename)
        fig.write_html(path, include_plotlyjs="cdn", full_html=True)
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
            fig.suptitle("EDA Summary Dashboard", fontsize=18, fontweight="bold", y=1.01,color=ACCENT_PURPLE)

            for i, col in enumerate(all_cols):
                ax = axes[i]
                if col in num_cols:
                    sns.histplot(self.data[col].dropna(), kde=True, ax=ax, color=ACCENT_CYAN, line_kws={'lw': 2})
                    ax.set_title(f"[Numeric] {col}", fontsize=11)
                else:
                    top_vals = self.data[col].value_counts().head(10)
                    sns.barplot(x=top_vals.values, y=top_vals.index.astype(str), ax=ax,palette=COLORY)
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
            custom_cmap = sns.blend_palette(COLORY, as_cmap=True)
            sns.heatmap(
                missing_matrix.T,
                ax=ax,
                cmap=custom_cmap,
                cbar_kws={"label": "Missing (1) / Present (0)"},
                linewidths=0.3,
                yticklabels=missing_cols
            )
            ax.set_title("Missing Values Matrix", fontsize=15, color=ACCENT_CYAN)
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
            custom_cmap = sns.blend_palette(COLORY, as_cmap=True)
            sns.heatmap(
                corr,
                ax=ax,
                annot=True,
                fmt=".2f",
                cmap=custom_cmap,
                center=0,
                mask=mask,
                linewidths=1,
                square=True,
                cbar_kws={"shrink": 0.8}
            )
            ax.set_title("Correlation Heatmap", fontsize=15, fontweight="bold", color=ACCENT_PURPLE)
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
                color_discrete_sequence=COLORY,
                template="plotly_dark",
                hover_data=self.data.columns.tolist()
            )
            fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=1, color=BG_COLOR)))
            self._apply_dark_layout_plotly(fig)
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
                color_discrete_sequence=COLORY,
                title=f"3D Scatter: {col1} / {col2} / {col3}",
                opacity=0.75,
                template="plotly_dark"
            )
            fig.update_traces(marker=dict(size=4),line=dict(width=1, color=BG_COLOR))
            self._apply_dark_layout_plotly(fig)
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
            g = sns.jointplot(data=clean, x=col1, y=col2, kind=kind, height=8, color=ACCENT_CYAN, marginal_kws=dict(fill=True, color=ACCENT_PURPLE))
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
            ct.plot(kind="bar", stacked=True, ax=ax, colormap=custom_cmap, edgecolor="white", linewidth=0.5)
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
                cmap=custom_cmap,
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
                palette="viridis",
                inner="box"
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

            # Limit categories to avoid huge figures
            cat_vals = self.data[cat_col].dropna().unique()
            max_cats = 6
            if len(cat_vals) > max_cats:
                cat_vals = cat_vals[:max_cats]
            df_filtered = self.data[self.data[cat_col].isin(cat_vals)]

            # Cap numeric columns to avoid excessive columns
            num_cols = num_cols[:6]

            # Melt numeric columns into long format for FacetGrid
            melted = df_filtered[num_cols + [cat_col]].melt(
                id_vars=cat_col, var_name="Variable", value_name="Value"
            )

            n_cols = min(len(num_cols), 3)
            n_rows = len(cat_vals)
            fig_w  = max(5, n_cols * 4.5)
            fig_h  = max(3, n_rows * 3.0)

            g = sns.FacetGrid(
                melted,
                col="Variable",
                row=cat_col,
                height=3.0,
                aspect=fig_w / max(fig_h, 1),
                sharey=False,
                col_wrap=None
            )
            g.map_dataframe(sns.histplot, x="Value", kde=True, color="steelblue", alpha=0.7)
            g.set_titles(col_template="{col_name}", row_template="{row_name}", size=10)
            g.set_axis_labels("Value", "Count", fontsize=9)
            g.fig.suptitle(
                f"Facet Grid by {cat_col}",
                fontsize=12, fontweight="bold", y=1.02
            )
            g.fig.tight_layout()

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
                color_discrete_sequence=PALETTE,
                title=f"Bubble Chart: {x} vs {y} (size={size}" + (f", color={color}" if color else "") + ")",
                template="plotly_dark",
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
    
    # //////////////////////////////////////////////////
   # ================================================================
    #                  AUTOMATIC DASHBOARD
    # ================================================================

    def generate_automatic_dashboard(self, save: bool = True) -> str:
        """
        Generates a comprehensive automatic EDA dashboard with as many
        relevant plots as possible, grouped into sections.
        """
        try:
            print("🚀 Generating Automatic EDA Dashboard...")

            plots_generated = []   # list of (section, title, path, size)
            # size: "wide" | "half" | "third"

            num_cols = self._get_cols_by_type(self._NUM)
            cat_cols = self._get_cols_by_type(self._CAT)

            # ── 1. OVERVIEW ────────────────────────────────────────────────────
            try:
                p = self.generate_summary_dashboard(save=True)
                if p: plots_generated.append(("overview", "📊 Summary Overview", p, "wide"))
            except Exception as e:
                print(f"Warning: summary dashboard: {e}")

            try:
                p = self.plot_correlation_heatmap(save=True)
                if p: plots_generated.append(("overview", "🔥 Correlation Heatmap", p, "half"))
            except: pass

            try:
                p = self.plot_missing_values_matrix(save=True)
                if p: plots_generated.append(("overview", "❓ Missing Values Matrix", p, "half"))
            except: pass

            # ── 2. NUMERIC DISTRIBUTIONS ────────────────────────────────────────
            for col in num_cols[:8]:
                try:
                    p = self._plot_single_distribution(col)
                    if p: plots_generated.append(("numeric", f"📈 Distribution: {col}", p, "third"))
                except: pass

            # ── 3. NUMERIC RELATIONSHIPS ────────────────────────────────────────
            if len(num_cols) >= 2:
                # Scatter pairs (up to 6)
                pairs = [(num_cols[i], num_cols[j])
                         for i in range(len(num_cols))
                         for j in range(i+1, len(num_cols))]
                for col1, col2 in pairs[:6]:
                    try:
                        p = self.plot_scatter_2d(col1, col2, save=True)
                        if p: plots_generated.append(("numeric", f"🔵 Scatter: {col1} vs {col2}", p, "half"))
                    except: pass

                # Joint plot (first pair)
                try:
                    p = self.plot_joint_plot(num_cols[0], num_cols[1], save=True)
                    if p: plots_generated.append(("numeric", f"🔗 Joint: {num_cols[0]} & {num_cols[1]}", p, "half"))
                except: pass

                # 3D scatter if 3+ numeric cols
                if len(num_cols) >= 3:
                    try:
                        p = self.plot_scatter_3d(num_cols[0], num_cols[1], num_cols[2], save=True)
                        if p: plots_generated.append(("numeric", f"🌐 3D Scatter", p, "half"))
                    except: pass

            # ── 4. CATEGORICAL OVERVIEWS ─────────────────────────────────────────
            for col in cat_cols[:6]:
                try:
                    p = self._plot_single_categorical(col)
                    if p: plots_generated.append(("categorical", f"🏷️ {col} Distribution", p, "third"))
                except: pass

            # Stacked bar (first two cat cols)
            if len(cat_cols) >= 2:
                for i in range(min(3, len(cat_cols)-1)):
                    try:
                        p = self.plot_stacked_bar(cat_cols[i], cat_cols[i+1], save=True)
                        if p: plots_generated.append(("categorical", f"📊 Stacked: {cat_cols[i]} × {cat_cols[i+1]}", p, "half"))
                    except: pass

                try:
                    p = self.plot_cross_tabulation(cat_cols[0], cat_cols[1], save=True)
                    if p: plots_generated.append(("categorical", f"🗂️ Cross-Tab: {cat_cols[0]} × {cat_cols[1]}", p, "half"))
                except: pass

            # ── 5. MIXED — NUM × CAT ────────────────────────────────────────────
            if num_cols and cat_cols:
                # Violin for each num × first cat (up to 4)
                for num_col in num_cols[:4]:
                    try:
                        p = self.plot_violin_plot_by_category(num_col, cat_cols[0], save=True)
                        if p: plots_generated.append(("mixed", f"🎻 Violin: {num_col} by {cat_cols[0]}", p, "half"))
                    except: pass

                # Facet grid (first 3 num cols × first cat)
                if len(num_cols) >= 2:
                    try:
                        facet_nums = num_cols[:min(3, len(num_cols))]
                        p = self.plot_facet_grid(facet_nums, cat_cols[0], save=True)
                        if p: plots_generated.append(("mixed", f"⊞ Facet Grid by {cat_cols[0]}", p, "wide"))
                    except: pass

                # Bubble chart (3 num cols needed)
                if len(num_cols) >= 3:
                    try:
                        p = self.plot_bubble_chart(num_cols[0], num_cols[1], num_cols[2],
                                                    color=cat_cols[0] if cat_cols else None, save=True)
                        if p: plots_generated.append(("mixed", f"🫧 Bubble Chart", p, "half"))
                    except: pass

            print(f"✅ Automatic Dashboard: {len(plots_generated)} plots generated.")

            if save and plots_generated:
                return self._create_dashboard_html(plots_generated)
            return None

        except Exception as e:
            raise Exception(f"Error in automatic dashboard: {e}")

    # ── helpers for single-column mini plots ────────────────────────────────
    def _plot_single_distribution(self, col: str) -> str:
        """Small histogram + KDE for one numeric column."""
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(self.data[col].dropna(), kde=True, ax=ax,
                     color=ACCENT_CYAN, line_kws={"lw": 2})
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        plt.tight_layout()
        fname = f"dist_{col}.png"
        return self._save_matplotlib(fig, fname)

    def _plot_single_categorical(self, col: str) -> str:
        """Small horizontal bar chart for one categorical column."""
        fig, ax = plt.subplots(figsize=(5, 3))
        top = self.data[col].value_counts().head(10)
        sns.barplot(x=top.values, y=top.index.astype(str), ax=ax, palette=COLORY)
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.set_xlabel("Count")
        plt.tight_layout()
        fname = f"cat_{col}.png"
        return self._save_matplotlib(fig, fname)

    def _create_dashboard_html(self, plots_list) -> str:
        """
        Professional analytics dashboard – compact multi-column grid with sections.
        plots_list items: (section, title, path, size)
          size: "wide" | "half" | "third"
        """
        # ── KPI stats ─────────────────────────────────────────────────────────
        num_cols      = self._get_cols_by_type(self._NUM)
        cat_cols      = self._get_cols_by_type(self._CAT)
        total_rows    = len(self.data)
        total_cols    = len(self.data.columns)
        missing_pct   = round(self.data.isnull().mean().mean() * 100, 1)
        num_count     = len(num_cols)
        cat_count     = len(cat_cols)
        dup_count     = int(self.data.duplicated().sum())
        complete_pct  = round((1 - self.data.isnull().mean().mean()) * 100, 1)

        kpi_defs = [
            ("📋", "Total Records",    f"{total_rows:,}",      ACCENT_CYAN),
            ("🔢", "Total Features",   str(total_cols),         "#818cf8"),
            ("📊", "Numeric Cols",     str(num_count),          ACCENT_CYAN),
            ("🏷️", "Categorical Cols", str(cat_count),          ACCENT_PURPLE),
            ("✅", "Completeness",     f"{complete_pct}%",     "#06d49d" if complete_pct >= 95 else "#f59e0b"),
            ("♊", "Duplicates",        str(dup_count),          "#ef4444" if dup_count > 0 else "#06d49d"),
            ("📉", "Missing %",        f"{missing_pct}%",      "#f59e0b" if missing_pct > 0 else "#06d49d"),
            ("🗂️", "Visualizations",  str(len(plots_list)),   ACCENT_PURPLE),
        ]

        kpi_html = ""
        for icon, label, value, color in kpi_defs:
            kpi_html += f"""<div class="kpi-card" style="--accent:{color};">
  <div class="kpi-top"><span class="kpi-icon">{icon}</span></div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-label">{label}</div>
  <div class="kpi-glow"></div>
</div>"""

        # ── Section order & labels ─────────────────────────────────────────────
        section_meta = {
            "overview":    ("🔍", "Dataset Overview"),
            "numeric":     ("📈", "Numeric Analysis"),
            "categorical": ("🏷️",  "Categorical Analysis"),
            "mixed":       ("🔀", "Mixed — Numeric × Categorical"),
        }

        # Group plots by section, preserving insertion order
        from collections import OrderedDict
        sections = OrderedDict()
        for item in plots_list:
            sec, title, path, size = item
            sections.setdefault(sec, []).append((title, path, size))

        # ── Build chart grid per section ───────────────────────────────────────
        sections_html = ""
        for sec_key, items in sections.items():
            icon, sec_label = section_meta.get(sec_key, ("📌", sec_key.title()))

            cards_html = ""
            for title, path, size in items:
                ext      = path.split('.')[-1].lower()
                filename = os.path.basename(path)
                safe_title = title.replace("'", "\\'")

                if size == "wide":
                    col_class = "span3"
                    h_img, h_iframe = "340px", "420px"
                elif size == "half":
                    col_class = "span1h"       # spans 1.5 cols → use span of 1 in 3-col
                    h_img, h_iframe = "260px", "320px"
                else:   # "third"
                    col_class = "span1"
                    h_img, h_iframe = "220px", "260px"

                if ext == "html":
                    media = (f'<iframe src="/phase2/view/{filename}" '
                             f'style="width:100%;height:{h_iframe};border:none;border-radius:8px;'
                             f'background:#0b0b18;" loading="lazy"></iframe>')
                else:
                    media = (f'<img src="/phase2/view/{filename}" alt="{title}" '
                             f'style="width:100%;height:{h_img};object-fit:contain;border-radius:8px;'
                             f'cursor:zoom-in;background:#0b0b18;" loading="lazy" '
                             f'onclick="openLb(this.src,\'{safe_title}\')">')

                cards_html += f"""<div class="chart-card {col_class}">
  <div class="chart-hdr">
    <span class="chart-dot"></span>
    <span class="chart-title">{title}</span>
    <a href="/phase2/view/{filename}" target="_blank" class="chart-ext" title="Open full">⤢</a>
  </div>
  <div class="chart-body">{media}</div>
</div>"""

            sections_html += f"""<section class="db-section">
  <div class="sec-label"><span>{icon} {sec_label}</span></div>
  <div class="chart-grid">{cards_html}</div>
</section>"""

        # ── Full HTML ──────────────────────────────────────────────────────────
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BRight AI — EDA Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#0b0b18;--card:#10112a;--card2:#0e0f26;
  --border:#1c1e45;--border2:#252760;
  --purple:{ACCENT_PURPLE};--cyan:{ACCENT_CYAN};
  --text:#dde1ff;--dim:#5c628a;
  --font:'Inter',sans-serif;
}}
html,body{{background:var(--bg);color:var(--text);font-family:var(--font);font-size:13px;min-height:100vh}}

/* ═══ HEADER ═══════════════════════════════════════════════════════════ */
.db-header{{
  display:flex;align-items:center;justify-content:space-between;
  padding:14px 24px;
  background:linear-gradient(135deg,#080918 0%,#0e0f2e 60%,#120f2e 100%);
  border-bottom:1px solid var(--border2);
  position:sticky;top:0;z-index:100;
  box-shadow:0 2px 20px rgba(0,0,0,.5);
}}
.hdr-left{{display:flex;align-items:center;gap:14px}}
.db-logo{{font-size:20px;font-weight:700;letter-spacing:.5px;color:var(--text)}}
.db-logo span{{
  background:linear-gradient(135deg,var(--cyan),var(--purple));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}}
.db-tagline{{font-size:10px;color:var(--dim);letter-spacing:2px;text-transform:uppercase;margin-top:2px}}
.hdr-chips{{display:flex;gap:8px}}
.hdr-chip{{
  padding:4px 12px;border-radius:20px;font-size:10px;font-weight:600;letter-spacing:.8px;
  border:1px solid;
}}
.chip-cyan{{background:rgba(6,212,157,.1);border-color:var(--cyan);color:var(--cyan)}}
.chip-purple{{background:rgba(168,85,247,.1);border-color:var(--purple);color:var(--purple)}}

/* ═══ BODY ══════════════════════════════════════════════════════════════ */
.db-body{{padding:20px 22px 48px;max-width:1600px;margin:0 auto}}

/* ═══ KPI STRIP ═════════════════════════════════════════════════════════ */
.kpi-strip{{
  display:grid;
  grid-template-columns:repeat(8,1fr);
  gap:10px;margin-bottom:26px;
}}
@media(max-width:1200px){{.kpi-strip{{grid-template-columns:repeat(4,1fr)}}}}
@media(max-width:700px){{.kpi-strip{{grid-template-columns:repeat(2,1fr)}}}}
.kpi-card{{
  background:var(--card);border:1px solid var(--border);border-radius:12px;
  padding:14px 14px 12px;position:relative;overflow:hidden;
  transition:transform .18s,border-color .18s,box-shadow .18s;
  cursor:default;
}}
.kpi-card:hover{{
  transform:translateY(-3px);
  border-color:var(--accent);
  box-shadow:0 6px 24px color-mix(in srgb,var(--accent) 20%,transparent);
}}
.kpi-top{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px}}
.kpi-icon{{font-size:18px;line-height:1}}
.kpi-value{{font-size:22px;font-weight:700;color:var(--accent);line-height:1;margin-bottom:4px}}
.kpi-label{{font-size:10px;color:var(--dim);font-weight:500;letter-spacing:.4px}}
.kpi-glow{{
  position:absolute;bottom:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--accent),transparent);
  border-radius:0 0 12px 12px;
}}

/* ═══ SECTION ═══════════════════════════════════════════════════════════ */
.db-section{{margin-bottom:28px}}
.sec-label{{
  display:flex;align-items:center;gap:10px;
  font-size:10px;font-weight:700;letter-spacing:3px;
  text-transform:uppercase;color:var(--dim);
  margin-bottom:12px;
}}
.sec-label span{{white-space:nowrap}}
.sec-label::after{{content:'';flex:1;height:1px;background:var(--border)}}

/* ═══ CHART GRID (3-column base) ════════════════════════════════════════ */
.chart-grid{{
  display:grid;
  grid-template-columns:repeat(3,1fr);
  gap:12px;
}}
@media(max-width:900px){{.chart-grid{{grid-template-columns:repeat(2,1fr)}}}}
@media(max-width:580px){{.chart-grid{{grid-template-columns:1fr}}}}

.chart-card{{
  background:var(--card);border:1px solid var(--border);border-radius:12px;
  overflow:hidden;transition:border-color .18s,box-shadow .18s;
  display:flex;flex-direction:column;
}}
.chart-card:hover{{
  border-color:rgba(168,85,247,.35);
  box-shadow:0 4px 22px rgba(168,85,247,.12);
}}
/* size modifiers */
.span3{{grid-column:1/-1}}
.span1h{{grid-column:span 1}}          /* plain half in 3-col = 1 col */
@media(min-width:900px){{
  .span1h{{grid-column:span 2}}        /* on wide screens span 2 of 3 */
}}
.span1{{grid-column:span 1}}

.chart-hdr{{
  display:flex;align-items:center;gap:8px;
  padding:10px 14px 9px;
  border-bottom:1px solid var(--border);
  background:rgba(255,255,255,.018);
  flex-shrink:0;
}}
.chart-dot{{
  width:7px;height:7px;border-radius:50%;flex-shrink:0;
  background:linear-gradient(135deg,var(--cyan),var(--purple));
}}
.chart-title{{flex:1;font-size:11px;font-weight:600;color:var(--text);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.chart-ext{{
  color:var(--dim);text-decoration:none;font-size:14px;line-height:1;
  padding:2px 5px;border-radius:5px;transition:color .15s,background .15s;
  flex-shrink:0;
}}
.chart-ext:hover{{color:var(--cyan);background:rgba(6,212,157,.1)}}
.chart-body{{padding:10px;flex:1;display:flex;align-items:center;justify-content:center}}
.chart-body img,.chart-body iframe{{border-radius:6px;display:block}}

/* ═══ LIGHTBOX ═══════════════════════════════════════════════════════════ */
#lb{{
  display:none;position:fixed;inset:0;z-index:9999;
  background:rgba(0,0,0,.92);backdrop-filter:blur(12px);
  align-items:center;justify-content:center;cursor:zoom-out;
}}
#lb.open{{display:flex}}
#lb-inner{{
  position:relative;max-width:94vw;max-height:90vh;
  display:flex;flex-direction:column;align-items:center;gap:12px;
  cursor:default;
}}
#lb-title{{
  font-size:11px;letter-spacing:2px;text-transform:uppercase;
  color:var(--cyan);text-align:center;
}}
#lb img{{
  max-width:92vw;max-height:82vh;border-radius:12px;
  box-shadow:0 0 60px rgba(168,85,247,.3);
}}
#lb-close{{
  position:fixed;top:16px;right:20px;
  background:rgba(239,68,68,.15);border:1px solid rgba(239,68,68,.4);
  color:#ef4444;width:34px;height:34px;border-radius:9px;
  font-size:16px;cursor:pointer;display:flex;align-items:center;justify-content:center;
  transition:background .15s;
}}
#lb-close:hover{{background:rgba(239,68,68,.3)}}

/* ═══ FOOTER ══════════════════════════════════════════════════════════ */
.db-footer{{
  text-align:center;padding:18px 24px;
  border-top:1px solid var(--border);
  color:var(--dim);font-size:10px;letter-spacing:1px;
}}
.db-footer strong{{color:var(--purple)}}
</style>
</head>
<body>

<header class="db-header">
  <div class="hdr-left">
    <div>
      <div class="db-logo">BRight <span>AI</span></div>
      <div class="db-tagline">Automatic EDA Dashboard</div>
    </div>
  </div>
  <div class="hdr-chips">
    <span class="hdr-chip chip-cyan">⬡ {total_rows:,} rows</span>
    <span class="hdr-chip chip-purple">◈ {total_cols} features</span>
    <span class="hdr-chip chip-cyan">📊 {len(plots_list)} charts</span>
  </div>
</header>

<div class="db-body">

  <!-- KPI STRIP -->
  <div class="kpi-strip">
    {kpi_html}
  </div>

  <!-- CHART SECTIONS -->
  {sections_html}

</div>

<footer class="db-footer">
  Generated by <strong>BRight AI</strong> · {total_rows:,} records · {total_cols} features · {len(plots_list)} visualizations
</footer>

<!-- LIGHTBOX -->
<div id="lb" onclick="closeLb()">
  <div id="lb-inner" onclick="event.stopPropagation()">
    <div id="lb-title"></div>
    <img id="lb-img" src="" alt="">
  </div>
  <button id="lb-close" onclick="closeLb()">✕</button>
</div>

<script>
function openLb(src,title){{
  document.getElementById('lb-img').src=src;
  document.getElementById('lb-title').textContent=title;
  document.getElementById('lb').classList.add('open');
  document.body.style.overflow='hidden';
}}
function closeLb(){{
  document.getElementById('lb').classList.remove('open');
  document.getElementById('lb-img').src='';
  document.body.style.overflow='';
}}
document.addEventListener('keydown',e=>{{if(e.key==='Escape')closeLb()}});
</script>
</body>
</html>"""

        dashboard_path = os.path.join(self.plots_dir, "automatic_dashboard.html")
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"📄 Dashboard saved: {dashboard_path}")
        return dashboard_path