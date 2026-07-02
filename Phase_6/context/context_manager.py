"""
Session-scoped chat context.

IMPORTANT: there is NO global singleton here anymore. Each user/session
must have its OWN ChatContext instance (created via ChatContext(), or
restored from disk via load_context()). This is what keeps two users'
datasets/questions from bleeding into each other.

Typical usage inside a Flask route:

    ctx = get_chat_context()          # loads this user's context (app.py helper)
    ctx.update_raw_dataset(df)
    save_chat_context(ctx)            # persists it back for next request
"""

import json
import os
import uuid

from Phase_6.context.dataset_context import DatasetContext
from Phase_6.context.eda_context import EDAContext
from Phase_6.context.visualization_context import VisualizationContext
from Phase_6.context.model_context import ModelContext
from Phase_6.context.prediction_context import PredictionContext
from Phase_6.context.conversation_context import ConversationContext
from Phase_6.context.nlp_context import NLPContext


class ChatContext:
    """Everything the chatbot can be asked about, for ONE session."""

    def __init__(self):
        self.dataset_context = DatasetContext()
        self.eda_context = EDAContext()
        self.visualization_context = VisualizationContext()
        self.model_context = ModelContext()
        self.prediction_context = PredictionContext()
        self.conversation_context = ConversationContext()
        self.nlp_context = NLPContext()

    # ── mutators ──────────────────────────────────────────────────
    def update_raw_dataset(self, df):
        """Call right after upload, BEFORE cleaning."""
        self.dataset_context.update_raw(df)

    def update_clean_dataset(self, df):
        """Call right after the preprocessing pipeline finishes."""
        self.dataset_context.update_clean(df)

    def log_eda(self, step):
        self.eda_context.add_step(step)

    def log_visualization(self, chart_name, description):
        self.visualization_context.add_chart(chart_name, description)

    def log_model(self, **kwargs):
        self.model_context.update(**kwargs)

    def log_prediction(self, prediction):
        self.prediction_context.add_prediction(prediction)

    def log_nlp(self, text_column, description):
        self.nlp_context.add_analysis(text_column, description)

    def add_conversation_turn(self, question, answer):
        self.conversation_context.add_turn(question, answer)

    # ── (de)serialization, so it can be saved to / loaded from disk ─
    def to_dict(self):
        return {
            "dataset": self.dataset_context.get_context(),
            "eda": self.eda_context.get_context(),
            "visualization": self.visualization_context.get_context(),
            "model": self.model_context.get_context(),
            "prediction": self.prediction_context.get_context(),
            "conversation": self.conversation_context.get_context(),
            "nlp": self.nlp_context.get_context(),
        }

    @classmethod
    def from_dict(cls, data: dict):
        ctx = cls()
        dataset = data.get("dataset", {}) or {}
        ctx.dataset_context.raw = dataset.get("raw", {}) or {}
        ctx.dataset_context.clean = dataset.get("clean", {}) or {}
        ctx.eda_context.logs = data.get("eda", []) or []
        ctx.visualization_context.charts = data.get("visualization", []) or []
        ctx.model_context.data = data.get("model", {}) or {}
        ctx.prediction_context.history = data.get("prediction", []) or []
        ctx.conversation_context.turns = data.get("conversation", []) or []
        ctx.nlp_context.analyses = data.get("nlp", []) or []
        return ctx


# ── Disk persistence helpers ─────────────────────────────────────────
# Same idiom the rest of the app already uses for DataFrames/schemas:
# write a file, keep only its PATH in the Flask session.

def new_context_filename() -> str:
    return f"chatctx_{uuid.uuid4().hex}.json"


def save_context(ctx: ChatContext, path: str):
    with open(path, "w", encoding="utf-8") as f:
        # default=str: safely handles numpy/pandas scalar types that
        # aren't natively JSON serializable (int64, Timestamp, ...).
        json.dump(ctx.to_dict(), f, default=str, ensure_ascii=False)


def load_context(path: str) -> ChatContext:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ChatContext.from_dict(data)
    return ChatContext()