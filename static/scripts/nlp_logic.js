/* ============================================================
   BRight AI — Phase 3: NLP Logic
   ============================================================ */

const NLP = { textColumns: [], allColumns: [] };

async function initNLP() {
    document.getElementById("mode-title").textContent =
        APP_MODE === "developer"
            ? "Developer Mode — full control over vectorization & classification"
            : "User Mode — one-click text analysis";

    try {
        const res = await fetch("/nlp/status");
        const data = await res.json();

        if (data.status !== "ready") {
            document.getElementById("nlp-no-data").style.display = "block";
            return;
        }

        NLP.textColumns = data.text_columns;
        NLP.allColumns = data.all_text_like_columns;

        document.getElementById("nlp-workspace").style.display = "block";
        document.getElementById("nlp-meta-rows").textContent = `${data.rows} rows`;

        const select = document.getElementById("nlp-text-column");
        const candidates = NLP.textColumns.length ? NLP.textColumns : NLP.allColumns;
        select.innerHTML = candidates.map(c => `<option value="${c}">${c}</option>`).join("");

        const hint = document.getElementById("nlp-detected-hint");
        if (NLP.textColumns.length) {
            hint.innerHTML = `<span class="col-chip categorical">✓ Auto-detected free-text column${NLP.textColumns.length > 1 ? "s" : ""}: ${NLP.textColumns.join(", ")}</span>`;
        } else {
            hint.innerHTML = `<span class="col-chip">No obvious free-text column detected — pick one manually above.</span>`;
        }

        if (APP_MODE === "developer") {
            document.getElementById("nlp-developer-mode").style.display = "block";
            const labelSelect = document.getElementById("nlp-dev-label");
            labelSelect.innerHTML =
                `<option value="">— None (rule-based sentiment) —</option>` +
                data.columns.filter(c => c !== select.value)
                    .map(c => `<option value="${c}">${c}</option>`).join("");
        } else {
            document.getElementById("nlp-user-mode").style.display = "block";
        }
    } catch (e) {
        showToast("Could not reach the server: " + e.message, "error");
    }
}

document.getElementById("nlp-auto-run-btn")?.addEventListener("click", () => runNLP(true));
document.getElementById("nlp-dev-run-btn")?.addEventListener("click", () => runNLP(false));

async function runNLP(auto) {
    const textColumn = document.getElementById("nlp-text-column").value;
    if (!textColumn) {
        showToast("Choose a text column first.", "error");
        return;
    }

    const loading = document.getElementById("nlp-loading");
    const results = document.getElementById("nlp-results");
    loading.classList.add("active");
    results.style.display = "none";

    const formData = new FormData();
    formData.append("text_column", textColumn);
    formData.append("auto", auto ? "true" : "false");

    if (!auto) {
        formData.append("method", document.getElementById("nlp-dev-method").value);
        formData.append("ngram_max", document.getElementById("nlp-dev-ngram").value);
        formData.append("top_n", document.getElementById("nlp-dev-topn").value);
        formData.append("classifier", document.getElementById("nlp-dev-classifier").value);
        const labelCol = document.getElementById("nlp-dev-label").value;
        if (labelCol) formData.append("label_column", labelCol);
    }

    try {
        const res = await fetch("/nlp/analyze", { method: "POST", body: formData });
        const data = await res.json();
        loading.classList.remove("active");

        if (data.status !== "success") {
            showToast(data.message || "NLP analysis failed.", "error");
            return;
        }

        renderNLPResults(data);
        results.style.display = "block";
        showToast("Analysis complete.", "success");
    } catch (e) {
        loading.classList.remove("active");
        showToast("Request failed: " + e.message, "error");
    }
}

function renderNLPResults(data) {
    const stats = data.statistics;
    const cards = document.getElementById("nlp-stat-cards");
    cards.innerHTML = `
        <div class="metric-card"><div class="metric-value">${stats.documents}</div><div class="metric-label">Documents</div></div>
        <div class="metric-card"><div class="metric-value">${stats.avg_word_count}</div><div class="metric-label">Avg Words</div></div>
        <div class="metric-card"><div class="metric-value">${stats.vocabulary_size}</div><div class="metric-label">Vocabulary Size</div></div>
        <div class="metric-card"><div class="metric-value">${stats.empty_documents}</div><div class="metric-label">Empty Docs</div></div>
    `;

    const plotsGrid = document.getElementById("nlp-plots-grid");
    plotsGrid.innerHTML = "";
    for (const [key, filename] of Object.entries(data.plots)) {
        plotsGrid.innerHTML += plotCard(key, `/nlp/view/${filename}`, `/nlp/download/${filename}`);
    }

    const sentiment = data.sentiment;
    const summaryPanel = document.getElementById("nlp-sentiment-summary");
    const body = document.getElementById("nlp-sentiment-body");
    summaryPanel.style.display = "block";

    if (sentiment.method === "lexicon") {
        const pills = Object.entries(sentiment.distribution_pct)
            .map(([label, pct]) => `<span class="sentiment-pill ${label}">${label}: ${pct}%</span>`).join("");
        body.innerHTML = `
            <p style="color:var(--txt-dim); font-size:13px; margin-bottom:6px;">
                Rule-based lexicon sentiment (no label column used) — dominant: <b style="color:var(--neon-c);">${sentiment.dominant_sentiment}</b>
            </p>
            <div class="sentiment-pill-row">${pills}</div>
        `;
    } else {
        body.innerHTML = `
            <div class="metric-row">
                <div class="metric-card"><div class="metric-value">${(sentiment.accuracy * 100).toFixed(1)}%</div><div class="metric-label">Accuracy</div></div>
                <div class="metric-card"><div class="metric-value">${sentiment.f1_score}</div><div class="metric-label">F1 Score</div></div>
                <div class="metric-card"><div class="metric-value">${sentiment.precision}</div><div class="metric-label">Precision</div></div>
                <div class="metric-card"><div class="metric-value">${sentiment.recall}</div><div class="metric-label">Recall</div></div>
            </div>
            <p style="color:var(--txt-dim); font-size:12.5px;">Trained ${sentiment.classifier} on ${sentiment.train_size} docs, tested on ${sentiment.test_size}.</p>
        `;
    }

    const keywordsTags = data.keywords.slice(0, 15)
        .map(k => `<span class="keyword-tag">${k.term}</span>`).join("");
    body.innerHTML += `<div style="margin-top:14px;"><b style="font-size:12.5px; color:var(--txt-dim);">Top keywords:</b><div class="keyword-tags">${keywordsTags}</div></div>`;
}

function plotCard(title, viewUrl, downloadUrl) {
    const niceTitle = title.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
    return `
        <div class="viz-result-card">
            <div class="result-header">
                <span class="result-title">${niceTitle}</span>
                <div class="result-actions">
                    <a class="result-action-btn view-btn" href="${viewUrl}" target="_blank"><i class="fas fa-expand"></i></a>
                    <a class="result-action-btn download-btn" href="${downloadUrl}" download><i class="fas fa-download"></i></a>
                </div>
            </div>
            <img class="plot-preview-img" src="${viewUrl}" loading="lazy" alt="${niceTitle}">
        </div>
    `;
}

function showToast(message, type = "success") {
    let toast = document.getElementById("nlp-toast");
    if (!toast) {
        toast = document.createElement("div");
        toast.id = "nlp-toast";
        document.body.appendChild(toast);
    }
    toast.className = `viz-toast ${type}`;
    const icon = type === "success" ? "fa-check-circle" : "fa-exclamation-circle";
    toast.innerHTML = `<i class="fas ${icon}"></i> ${message}`;
    requestAnimationFrame(() => toast.classList.add("show"));
    clearTimeout(toast._hideTimeout);
    toast._hideTimeout = setTimeout(() => toast.classList.remove("show"), 3500);
}

window.onload = initNLP;
