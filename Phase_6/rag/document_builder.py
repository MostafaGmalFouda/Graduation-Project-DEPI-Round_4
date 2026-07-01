from langchain_core.documents import Document


def _dataset_documents(dataset_context):
    documents = []
    dataset = dataset_context.get_context()

    stages = [
        ("raw", "Raw data (BEFORE cleaning)"),
        ("clean", "Clean data (AFTER cleaning)"),
    ]

    for key, label in stages:
        stage_data = dataset.get(key)
        if not stage_data:
            continue

        missing = stage_data.get("missing_values", {}) or {}
        missing_total = sum(missing.values()) if missing else 0
        if missing_total == 0:
            missing_text = (
                f"[{label}] Missing values: NONE. Every single column has "
                f"0 (zero) missing values. There are NO missing values anywhere in this data."
            )
        else:
            offenders = [f"{col} has {cnt} missing value(s)" for col, cnt in missing.items() if cnt > 0]
            clean_cols = [col for col, cnt in missing.items() if cnt == 0]
            missing_text = (
                f"[{label}] Missing values: total {missing_total} missing cell(s) in the dataset. "
                f"Columns WITH missing values: {'; '.join(offenders)}. "
                f"Columns with 0 (NO) missing values: {', '.join(clean_cols) if clean_cols else 'none'}."
            )

        dup_count = stage_data.get("duplicates", 0)
        if dup_count == 0:
            dup_text = f"[{label}] Duplicate rows: NONE. There are 0 (zero) duplicate rows in this data."
        else:
            dup_text = f"[{label}] Duplicate rows: {dup_count} duplicate row(s) found in this data."

        documents.extend([
            Document(
                page_content=f"[{label}] Number of rows: {stage_data['rows']}",
                metadata={"source": "dataset", "stage": key, "section": "rows"},
            ),
            Document(
                page_content=f"[{label}] Number of columns: {stage_data['columns']}",
                metadata={"source": "dataset", "stage": key, "section": "columns"},
            ),
            Document(
                page_content=f"[{label}] Columns:\n{stage_data['column_names']}",
                metadata={"source": "dataset", "stage": key, "section": "column_names"},
            ),
            Document(
                page_content=f"[{label}] Data Types:\n{stage_data['dtypes']}",
                metadata={"source": "dataset", "stage": key, "section": "dtypes"},
            ),
            Document(
                page_content=missing_text,
                metadata={"source": "dataset", "stage": key, "section": "missing_values"},
            ),
            Document(
                page_content=dup_text,
                metadata={"source": "dataset", "stage": key, "section": "duplicates"},
            ),
            Document(
                page_content=f"[{label}] Statistics:\n{stage_data['describe']}",
                metadata={"source": "dataset", "stage": key, "section": "statistics"},
            ),
        ])

    return documents


def build_documents(chat_context):
    """
    Build the retrievable document set for ONE session's ChatContext.
    `chat_context` is a Phase_6.context.context_manager.ChatContext instance.
    """
    documents = []

    # Dataset (raw + clean, both queryable)
    documents.extend(_dataset_documents(chat_context.dataset_context))

    # EDA / cleaning steps log
    for step in chat_context.eda_context.get_context():
        documents.append(
            Document(page_content=step, metadata={"source": "eda"})
        )

    # Visualizations
    for chart in chat_context.visualization_context.get_context():
        documents.append(
            Document(
                page_content=(
                    f"Chart Name:\n{chart['chart']}\n\n"
                    f"Description:\n{chart['description']}"
                ),
                metadata={"source": "visualization", "chart": chart["chart"]},
            )
        )

    # Model info (optional)
    model_data = chat_context.model_context.get_context()
    if model_data:
        documents.append(
            Document(
                page_content=f"Model info:\n{model_data}",
                metadata={"source": "model"},
            )
        )

    # Predictions history (optional)
    for i, prediction in enumerate(chat_context.prediction_context.get_context()):
        documents.append(
            Document(
                page_content=f"Prediction #{i + 1}:\n{prediction}",
                metadata={"source": "prediction"},
            )
        )

    return documents