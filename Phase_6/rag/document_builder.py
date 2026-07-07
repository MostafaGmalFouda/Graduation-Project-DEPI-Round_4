from langchain_core.documents import Document


def build_guaranteed_facts(chat_context) -> str:
    """
    Core structural facts (row/column counts, missing values, duplicates)
    for both dataset stages. These are returned as a fixed text block that
    ALWAYS goes into the prompt — they do NOT depend on similarity search
    finding the right document among k results. Precise counting questions
    ("how many rows?", "any missing values?") must never be left to chance
    retrieval; they're small enough to just always include.
    """
    dataset = chat_context.dataset_context.get_context()
    lines = []

    for key, label in (("raw", "Raw data (BEFORE cleaning)"), ("clean", "Clean data (AFTER cleaning)")):
        stage = dataset.get(key)
        if not stage:
            continue
        missing = stage.get("missing_values", {}) or {}
        missing_total = sum(missing.values()) if missing else 0
        lines.append(
            f"[{label}] rows={stage['rows']}, columns={stage['columns']}, "
            f"duplicate_rows={stage.get('duplicates', 0)}, "
            f"total_missing_values={missing_total}, "
            f"column_names={stage['column_names']}"
        )

    return "\n".join(lines) if lines else "No dataset has been loaded yet."


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
                metadata={"source": "dataset", "type": "dataset", "stage": key, "section": "rows", "id": f"dataset:{key}:rows"},
            ),
            Document(
                page_content=f"[{label}] Number of columns: {stage_data['columns']}",
                metadata={"source": "dataset", "type": "dataset", "stage": key, "section": "columns", "id": f"dataset:{key}:columns"},
            ),
            Document(
                page_content=f"[{label}] Columns:\n{stage_data['column_names']}",
                metadata={"source": "dataset", "type": "dataset", "stage": key, "section": "column_names", "id": f"dataset:{key}:column_names"},
            ),
            Document(
                page_content=f"[{label}] Data Types:\n{stage_data['dtypes']}",
                metadata={"source": "dataset", "type": "dataset", "stage": key, "section": "dtypes", "id": f"dataset:{key}:dtypes"},
            ),
            Document(
                page_content=missing_text,
                metadata={"source": "dataset", "type": "dataset", "stage": key, "section": "missing_values", "id": f"dataset:{key}:missing_values"},
            ),
            Document(
                page_content=dup_text,
                metadata={"source": "dataset", "type": "dataset", "stage": key, "section": "duplicates", "id": f"dataset:{key}:duplicates"},
            ),
            Document(
                page_content=f"[{label}] Statistics:\n{stage_data['describe']}",
                metadata={"source": "dataset", "type": "dataset", "stage": key, "section": "statistics", "id": f"dataset:{key}:statistics"},
            ),
        ])

    return documents


def build_documents(chat_context):
    """
    Build the retrievable document set for ONE session's ChatContext.
    `chat_context` is a Phase_6.context.context_manager.ChatContext instance.

    Every Document gets a consistent metadata shape:
        type    - which context this came from (dataset/eda/visualization/nlp/...)
        section - what kind of info within that type
        id      - a stable, human-readable identifier, useful for logging
                  which exact document got retrieved for a given answer
    """
    documents = []

    # Dataset (raw + clean, both queryable)
    documents.extend(_dataset_documents(chat_context.dataset_context))

    # EDA / cleaning steps log
    for i, step in enumerate(chat_context.eda_context.get_context()):
        documents.append(
            Document(
                page_content=step,
                metadata={"source": "eda", "type": "eda", "section": "step", "id": f"eda:{i}"},
            )
        )

    # Visualizations
    for i, chart in enumerate(chat_context.visualization_context.get_context()):
        documents.append(
            Document(
                page_content=(
                    f"Chart Name:\n{chart['chart']}\n\n"
                    f"Description:\n{chart['description']}"
                ),
                metadata={
                    "source": "visualization", "type": "visualization", "section": "chart",
                    "chart": chart["chart"], "id": f"viz:{i}:{chart['chart']}",
                },
            )
        )

    # NLP analyses — the statistical summary (keywords, sentiment, stats)
    for i, analysis in enumerate(chat_context.nlp_context.get_context()):
        documents.append(
            Document(
                page_content=(
                    f"NLP analysis of column '{analysis['text_column']}':\n"
                    f"{analysis['description']}"
                ),
                metadata={
                    "source": "nlp", "type": "nlp", "section": "analysis",
                    "text_column": analysis["text_column"], "id": f"nlp:{i}:{analysis['text_column']}",
                },
            )
        )

    # NLP raw text content — the ACTUAL text (reviews/comments/etc.), so the
    # chatbot can answer content questions ("what are people complaining
    # about?"), not just questions about the summary statistics above.
    for entry in chat_context.nlp_context.get_raw_texts():
        col = entry["text_column"]
        for i, text in enumerate(entry["texts"]):
            if not text or not text.strip():
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": "nlp_raw_text", "type": "nlp", "section": "raw_text",
                        "text_column": col, "row": i, "id": f"nlp_raw:{col}:{i}",
                    },
                )
            )

    # Model info (optional)
    model_data = chat_context.model_context.get_context()
    if model_data:
        documents.append(
            Document(
                page_content=f"Model info:\n{model_data}",
                metadata={"source": "model", "type": "model", "section": "info", "id": "model:info"},
            )
        )

    # Predictions history (optional)
    for i, prediction in enumerate(chat_context.prediction_context.get_context()):
        documents.append(
            Document(
                page_content=f"Prediction #{i + 1}:\n{prediction}",
                metadata={
                    "source": "prediction", "type": "prediction", "section": "history",
                    "id": f"prediction:{i}",
                },
            )
        )

    return documents