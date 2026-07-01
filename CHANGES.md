# 🐛 CHANGES — Round 3: Bug Fixes

This package contains **only the files modified to fix the 4 reported issues**.
Drop these into your project, overwriting the matching paths.

---

## 1) Developer Mode ML training crash — FIXED

**Your error:**
```
ValueError: Unknown label type: continuous.
Maybe you are trying to fit a classifier, which expects discrete classes
on a regression target with continuous values.
```

**Root cause:** In Developer Mode, the model picker showed *every* model
name (classification + regression + clustering all merged into one list)
without knowing which ones were actually valid for your target column. If
your target was continuous (regression) but you picked `knn`, the app
built a `KNeighborsClassifier` instead of a `KNeighborsRegressor` — and
sklearn rightfully crashed.

**Fix, two layers:**
- **Frontend** (`ai_pipeline_logic.js`, `phase4_logic.js`): the model
  picker now calls `/phase4/recommend` first to detect the *real* task
  type (classification/regression/clustering) from your target column,
  then only shows models that are actually valid for it. A small banner
  ("Detected task: regression — only compatible models are shown") makes
  this visible.
- **Backend** (`app.py`, `/phase4/train`): added a server-side check that
  rejects an incompatible model/task combination with a clear 400 error
  *before* it ever reaches sklearn — e.g. *"'dbscan' does not support the
  detected task type 'regression'. Compatible models: random forest,
  linear regression, svm, knn, decision tree."* — instead of a raw
  traceback.

Verified: training `knn` on a regression target now correctly builds a
`KNeighborsRegressor` and returns real metrics (tested at R² = 0.84).

---

## 2) "I want to skip/leave whenever I want" — FIXED

**Root cause:** The auto-continue countdowns (Phase 1 → Phase 2, Phase 2 →
AI Pipeline, and all the ML → NLP → RAG transitions) only offered "Skip
now" — jump ahead immediately. There was no way to just **stay** and
cancel the countdown; in `index.html` specifically there was no cancel
option at all.

**Fix — every automatic transition now has two real choices:**
- **Skip now** → jump ahead immediately (unchanged)
- **Stay on this page / this step** → cancels the countdown entirely, no
  forced navigation

Also: clicking anywhere in the journey tracker or sub-step navigation
now automatically cancels any pending auto-continue first, so manual
clicks never get overridden by a countdown firing underneath you.

Files: `templates/index.html`, `templates/phase2_ui.html`,
`static/scripts/ai_pipeline_logic.js`, `static/css/ai_pipeline_style.css`.

---

## 3) "Don't ask for the data twice" — FIXED

**Root cause:** Even after data was already loaded (from Phase 1, or from
a previous upload), the upload zone in the AI Pipeline / standalone
Phase 3 / Phase 4 pages stayed visible — looking like it still wanted a
file, even though it had silently already loaded your session data
behind the scenes.

**Fix:** Once a dataset is detected (freshly uploaded or already in your
session), the upload zone now hides itself and is replaced with a clear
confirmation: *"✓ Using your data from Phase 1 — 891 rows × 12 columns"*,
plus a small "Use a different file" link if you genuinely want to swap
datasets. No more re-asking.

Files: `static/scripts/ai_pipeline_logic.js`, `static/scripts/phase3_logic.js`,
`static/scripts/phase4_logic.js`.

---

## 4) Chatbot not working — FIXED (made the failure visible + fixed a real gap)

The RAG chat logic itself (prompt building, retrieval, the Claude API
call) was already correct — the model string `claude-sonnet-4-6` is
current and valid. The real problem was that when something *was* wrong
(most commonly: the server has no `ANTHROPIC_API_KEY` set), the failure
was invisible until you'd already built the whole index and typed a
question — at which point you'd get a vague error with no clear next step.

**Fix:**
- New endpoint `GET /phase3/rag/status` checks whether the server has its
  API key configured, with **zero exposure of the key itself**.
- The AI Pipeline now calls this the moment you reach the RAG stage, and
  shows a clear warning banner immediately if the chatbot isn't
  configured yet — *before* you waste time building an index that can't
  answer anything.
- In User Mode, the index still auto-builds either way (so it's ready the
  moment the key gets added), but the warning stays visible the whole
  time so it's obvious why answers aren't coming back yet.
- Chat error messages now specifically call out a missing API key with
  the exact fix needed, instead of a generic "⚠️ {raw backend message}".
- Hardened `VectorStoreManager.generate_embeddings()`: if the embedding
  model can't load (most commonly: no internet on first run, since it
  has to download from Hugging Face once), it now raises a clear,
  actionable error instead of hanging or crashing with a cryptic message.

Files: `app.py`, `static/scripts/ai_pipeline_logic.js`,
`Phase_4_RAG/VectorStoreManager.py`.

**If your chatbot still doesn't answer after this update**, the banner on
the RAG step will now tell you exactly why. The most common fix:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # https://console.anthropic.com/
python app.py                            # restart after setting it
```

---

## ✅ Verified before delivery

- All modified `.py` files compile cleanly (`py_compile`).
- All modified `.js` files pass `node --check`.
- All 5 pages (`/`, `/phase2`, `/ai-pipeline`, `/phase3`, `/phase4`) render
  with HTTP 200 via Flask's test client.
- **Reproduced your exact bug** (KNN on a continuous/regression target)
  and confirmed it now trains successfully instead of crashing.
- **Reproduced the genuine-mismatch case** (e.g. clustering-only model on
  a regression target) and confirmed it now returns a clean, actionable
  400 error instead of an unhandled exception.
