import pandas as pd
from ydata_profiling import ProfileReport
from jinja2 import Template
import os
import io
import base64
from collections import Counter

# ── Optional NLP libraries ──────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    NLP_VIZ_AVAILABLE = True
except ImportError:
    NLP_VIZ_AVAILABLE = False
    print("[Warning] matplotlib or wordcloud not installed. "
          "NLP visualisations will be skipped.")

try:
    import nltk
    from nltk.corpus import stopwords
    # Check if stopwords are available; if not, set flag to False
    try:
        stopwords.words('english')
        NLTK_AVAILABLE = True
        _STOPWORDS = set(stopwords.words('english'))
    except LookupError:
        print("[Warning] nltk stopwords not found. Run nltk.download('stopwords') to enable text summarization.")
        NLTK_AVAILABLE = False
        _STOPWORDS = set()
except ImportError:
    NLTK_AVAILABLE = False
    _STOPWORDS = set()
    print("[Warning] nltk not installed. NLP text analysis features will be limited.")


class ReportGenerator:
    """
    Generates EDA reports with support for both structured and NLP data.

    Modes:
        - "basic"  : HTML report using a Jinja2 template.
        - "detailed": Full ydata_profiling report.

    NLP features require nltk, matplotlib, and wordcloud to be installed.
    """

    TEXT_AVG_WORD_THRESHOLD = 3

    def __init__(self, data: pd.DataFrame, target: str = None):
        self.data = data
        self.target = target
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self._stopwords = _STOPWORDS

    # ------------------------------------------------------------------
    # Basic summary methods
    # ------------------------------------------------------------------
    def summary_stats(self) -> pd.DataFrame:
        return self.data.describe(include='all')

    def correlation_matrix(self) -> pd.DataFrame:
        return self.data.corr(numeric_only=True)

    def missing_values(self) -> pd.DataFrame:
        missing = self.data.isnull().sum()
        percent = (missing / len(self.data)) * 100
        return pd.DataFrame({
            "Missing Count": missing,
            "Percentage (%)": percent
        })

    def insights(self) -> list:
        insights = []

        # ── Missing values ──
        # ✅ FIX: use index=False to avoid "too many values to unpack"
        for col, val in self.missing_values().itertuples(index=False):
            if val > 0:
                insights.append(f"Column '{col}' has {val:.2f}% missing values")

        # ── Strong correlations (top 5) ──
        corr = self.correlation_matrix()
        strong_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i):
                r = corr.iloc[i, j]
                if abs(r) > 0.8:
                    strong_pairs.append((corr.columns[i], corr.columns[j], r))
        strong_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for col1, col2, r in strong_pairs[:5]:
            insights.append(
                f"Strong correlation between '{col1}' and '{col2}' (r={r:.2f})"
            )

        # ── Duplicates ──
        dup = self.data.duplicated().sum()
        if dup:
            insights.append(f"Dataset contains {dup} duplicate rows")

        return insights

    def overview(self) -> dict:
        return {
            "Shape": self.data.shape,
            "Columns": list(self.data.columns),
            "Data Types": self.data.dtypes.astype(str),
        }

    # ------------------------------------------------------------------
    # NLP‑specific methods (used by EDAPipeline) – gracefully degrade if nltk missing
    # ------------------------------------------------------------------
    def feature_types(self) -> dict:
        """
        Classify each column into one of: 'numeric', 'categorical',
        'datetime', 'text', or 'id'.
        """
        types = {
            'numeric': [],
            'categorical': [],
            'datetime': [],
            'text': [],
            'id': []
        }

        for col in self.data.columns:
            dtype = self.data[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                types['numeric'].append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                types['datetime'].append(col)
            elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
                if NLTK_AVAILABLE and self._is_text_column(col):
                    types['text'].append(col)
                else:
                    n_unique = self.data[col].nunique(dropna=True)
                    if n_unique / len(self.data) > 0.9 or self._is_mixed_alphanum(col):
                        types['id'].append(col)
                    else:
                        types['categorical'].append(col)
            else:
                types['categorical'].append(col)

        return types

    def text_summary(self) -> str:
        """Generate a human‑readable summary for all text columns (only if nltk available)."""
        if not NLTK_AVAILABLE:
            return "NLP features disabled – install nltk and download stopwords to enable text summarization."

        text_cols = self.feature_types()['text']
        if not text_cols:
            return "No text columns found."

        lines = ["📝 TEXT COLUMN SUMMARY\n"]
        for col in text_cols:
            series = self.data[col].dropna().astype(str)
            if len(series) == 0:
                continue

            n_missing = self.data[col].isna().sum()
            empty_strs = (self.data[col].astype(str).str.strip() == "").sum()
            empty_ratio = (n_missing + empty_strs) / len(self.data)

            lengths = series.str.len()
            word_counts = series.str.split().str.len()

            lines.append(f"  • {col}:")
            lines.append(f"      - Empty/missing ratio : {empty_ratio:.2%}")
            lines.append(f"      - Avg character length: {lengths.mean():.1f}")
            lines.append(f"      - Avg word count      : {word_counts.mean():.1f}")
            lines.append(f"      - Max word count      : {word_counts.max()}")

            # Top words (excluding stopwords if available)
            all_words = (
                series.str.lower()
                .str.replace(r"[^\w\s]", "", regex=True)
                .str.split()
                .explode()
            )
            if self._stopwords:
                all_words = all_words[~all_words.isin(self._stopwords)]
            top_words = all_words.value_counts().head(10)
            if not top_words.empty:
                lines.append("      - Top 10 meaningful words:")
                for word, count in top_words.items():
                    lines.append(f"          • {word}: {count}")

            # Top bigrams
            def bigrams(tokens):
                return [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
            all_bigrams = series.str.lower().str.split().apply(bigrams).explode()
            top_bigrams = all_bigrams.value_counts().head(5)
            if not top_bigrams.empty:
                lines.append("      - Top 5 bigrams:")
                for bg, count in top_bigrams.items():
                    lines.append(f"          • {bg}: {count}")

            lines.append("")

        return "\n".join(lines)

    def nlp_readiness(self) -> str:
        """Assess readiness (only if nltk available)."""
        if not NLTK_AVAILABLE:
            return "NLP readiness check requires nltk with stopwords – install and download to enable."

        text_cols = self.feature_types()['text']
        if not text_cols:
            return "No text columns to assess."

        lines = ["🧹 NLP READINESS ASSESSMENT\n"]
        for col in text_cols:
            series = self.data[col].dropna().astype(str)
            if len(series) == 0:
                continue

            sample = series.head(100)

            has_url = sample.str.contains(r'https?://|www\.', regex=True).any()
            has_email = sample.str.contains(r'\S+@\S+', regex=True).any()
            has_numbers = sample.str.contains(r'\d+', regex=True).any()
            has_special = sample.str.contains(r'[^\w\s]', regex=True).any()

            all_words = " ".join(series).lower().split()
            total_words = len(all_words)
            if total_words > 0 and self._stopwords:
                stop_ratio = sum(1 for w in all_words if w in self._stopwords) / total_words
            else:
                stop_ratio = 0

            vocab = set(all_words)
            vocab_richness = len(vocab) / total_words if total_words > 0 else 0

            lines.append(f"  • {col}:")
            lines.append(f"      - Contains URLs       : {'Yes' if has_url else 'No'}")
            lines.append(f"      - Contains emails     : {'Yes' if has_email else 'No'}")
            lines.append(f"      - Contains numbers    : {'Yes' if has_numbers else 'No'}")
            lines.append(f"      - Contains special ch.: {'Yes' if has_special else 'No'}")
            lines.append(f"      - Stopword ratio      : {stop_ratio:.2%}")
            lines.append(f"      - Vocabulary richness : {vocab_richness:.2%}")

            issues = []
            if has_url:
                issues.append("URLs present → remove or replace with token")
            if has_email:
                issues.append("Emails present → anonymise or remove")
            if has_numbers:
                issues.append("Numbers present → consider removing or keeping based on context")
            if has_special:
                issues.append("Special characters present → clean with regex")
            if stop_ratio > 0.7:
                issues.append("High stopword ratio → remove stopwords")
            if vocab_richness < 0.01:
                issues.append("Very low vocabulary richness → text may be repetitive or too short")

            if issues:
                lines.append("      - ⚠️  Recommendations:")
                for issue in issues:
                    lines.append(f"          • {issue}")
            else:
                lines.append("      - ✅ Text looks clean and ready for NLP.")

            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helper methods for text detection
    # ------------------------------------------------------------------
    def _is_text_column(self, col: str) -> bool:
        series = self.data[col].dropna().astype(str)
        if len(series) == 0:
            return False
        avg_words = series.str.split().str.len().mean()
        return avg_words > self.TEXT_AVG_WORD_THRESHOLD

    def _is_mixed_alphanum(self, col: str) -> bool:
        sample = self.data[col].dropna().astype(str).head(50)
        if len(sample) == 0:
            return False
        mixed = sample.apply(
            lambda x: any(c.isalpha() for c in x) and any(c.isdigit() for c in x)
        ).sum()
        return (mixed / len(sample)) >= 0.3

    # ------------------------------------------------------------------
    # NLP visualisation generator (conditional) – requires matplotlib & wordcloud
    # ------------------------------------------------------------------
    def _generate_nlp_visualisations(self, text_cols: list) -> dict:
        """
        Generate word clouds, top-word bar charts, and length histograms
        for each text column.
        """
        if not NLP_VIZ_AVAILABLE:
            return {
                'has_viz': False,
                'error': "matplotlib/wordcloud not installed. Install with: pip install matplotlib wordcloud"
            }

        if not text_cols:
            return {'has_viz': False, 'error': None}

        images = []
        for col in text_cols:
            series = self.data[col].dropna().astype(str)
            if len(series) == 0:
                continue

            # Word frequencies (excluding stopwords if available)
            all_words = (
                series.str.lower()
                .str.replace(r"[^\w\s]", "", regex=True)
                .str.split()
                .explode()
            )
            if self._stopwords:
                all_words = all_words[~all_words.isin(self._stopwords)]
            freq = Counter(all_words)
            top_words = freq.most_common(20)

            if not top_words:
                continue

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f"📊 NLP Analysis: {col}", fontsize=14, y=1.02)

            # Plot A: Length distribution
            word_counts = series.str.split().str.len()
            axes[0].hist(word_counts, bins=30, color='skyblue', edgecolor='black')
            axes[0].set_title('Word Count Distribution')
            axes[0].set_xlabel('Number of words')
            axes[0].set_ylabel('Frequency')

            # Plot B: Top 20 words
            words, counts = zip(*top_words) if top_words else ([], [])
            axes[1].barh(words, counts, color='lightcoral')
            axes[1].set_title('Top 20 Meaningful Words')
            axes[1].set_xlabel('Frequency')
            axes[1].invert_yaxis()

            # Plot C: Word Cloud
            if len(freq) > 0:
                wc = WordCloud(
                    width=400, height=300,
                    background_color='white',
                    max_words=100,
                    colormap='viridis'
                ).generate_from_frequencies(freq)
                axes[2].imshow(wc, interpolation='bilinear')
                axes[2].axis('off')
                axes[2].set_title('Word Cloud')
            else:
                axes[2].text(0.5, 0.5, "Not enough words", ha='center', va='center')
                axes[2].axis('off')

            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            plt.close(fig)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()

            images.append({
                'column': col,
                'base64': img_base64
            })

        return {
            'has_viz': bool(images),
            'images': images,
            'error': None
        }

    # ------------------------------------------------------------------
    # Report generation methods
    # ------------------------------------------------------------------
    def manual_report(self) -> str:
        """Generate basic HTML report, including NLP visuals if applicable."""
        PROJECT_ROOT = os.path.abspath(os.path.join(self.BASE_DIR, os.pardir))
        reports_dir = os.path.join(PROJECT_ROOT, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        template_path = os.path.join(self.BASE_DIR, "template.html")
        if not os.path.exists(template_path):
            raise FileNotFoundError(
                f"Template file not found at {template_path}. "
                "Please ensure 'template.html' exists in the Phase_1 directory."
            )

        report = {
            "shape": self.data.shape,
            "total_missing": self.data.isnull().sum().sum(),
            "duplicates": self.data.duplicated().sum(),
            "summary": self.summary_stats().to_html(),
            "missing": self.missing_values().to_html(),
            "correlation": self.correlation_matrix().to_html(),
            "insights": self.insights(),
            "top_missing": self.missing_values()["Percentage (%)"].idxmax(),
            "top_missing_val": self.missing_values()["Percentage (%)"].max(),
        }

        # ── NLP detection and visualisation (CONDITIONAL) ──
        text_cols = self.feature_types()['text']
        if text_cols:
            print(f"[ReportGenerator] Detected text columns: {text_cols}. "
                  "Generating NLP visualisations...")
            viz_result = self._generate_nlp_visualisations(text_cols)
            report['has_nlp'] = True
            report['text_columns'] = text_cols
            report['nlp_viz'] = viz_result
        else:
            report['has_nlp'] = False
            report['text_columns'] = []
            report['nlp_viz'] = {'has_viz': False, 'images': [], 'error': None}

        with open(template_path, "r", encoding="utf-8") as f:
            template = Template(f.read())

        html = template.render(**report)

        output_path = os.path.join(reports_dir, "final_report.html")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return output_path

    def auto_report(self, file_name: str = "detailed_report.html", explorative: bool = True) -> str:
        PROJECT_ROOT = os.path.abspath(os.path.join(self.BASE_DIR, os.pardir))
        reports_dir = os.path.join(PROJECT_ROOT, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        output_path = os.path.join(reports_dir, file_name)
        profile = ProfileReport(self.data, explorative=explorative)
        profile.to_file(output_path)
        return output_path

    def generate_report(self, mode: str = "basic", file_name: str = "report.html") -> str:
        if mode == "basic":
            return self.manual_report()
        elif mode == "detailed":
            return self.auto_report(file_name=file_name)
        else:
            raise ValueError("mode must be 'basic' or 'detailed'")