/* ============================================================
   BRight AI — Phase 3: NLP Logic
   ============================================================ */

const NLP = { textColumns: [], allColumns: [] };

// Just flips which panel is visible + the mode-title text — both panels'
// controls (text column select, run buttons) are already fully wired up
// regardless of mode, so switching is instant and never loses anything.
function applyNLPModeUI(mode) {
    APP_MODE = mode;
    document.getElementById("mode-title").textContent =
        mode === "developer"
            ? "Developer Mode — full control over vectorization & classification"
            : "User Mode — one-click text analysis";
    document.getElementById("nlp-developer-mode").style.display = mode === "developer" ? "block" : "none";
    document.getElementById("nlp-user-mode").style.display = mode === "developer" ? "none" : "block";
}
window.applyPageMode = applyNLPModeUI;

async function initNLP() {
    applyNLPModeUI(APP_MODE);

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
    loading.style.display = "flex";
    loading.classList.add("active");
    results.style.display = "none";

    const formData = new FormData();
    formData.append("text_column", textColumn);
    formData.append("auto", auto ? "true" : "false");

    if (!auto) {
        formData.append("method", document.getElementById("nlp-dev-method").value);
        formData.append("ngram_max", document.getElementById("nlp-dev-ngram").value);
        formData.append("top_n", document.getElementById("nlp-dev-topn").value);
        const includeSentiment = document.getElementById("nlp-dev-sentiment").checked;
        formData.append("include_sentiment", includeSentiment ? "true" : "false");
        const includeTrigrams = document.getElementById("nlp-dev-trigrams").checked;
        formData.append("include_trigrams", includeTrigrams ? "true" : "false");
    }

    try {
        const res = await fetch("/nlp/analyze", { method: "POST", body: formData });
        const data = await res.json();
        loading.style.display = "none";
        loading.classList.remove("active");

        if (data.status !== "success") {
            showToast(data.message || "NLP analysis failed.", "error");
            return;
        }

        renderNLPResults(data);
        results.style.display = "block";
        showToast("Analysis complete.", "success");
    } catch (e) {
        loading.style.display = "none";
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

    const ba = data.before_after;
    const baPanel = document.getElementById("nlp-before-after-panel");
    const baBody = document.getElementById("nlp-before-after-body");
    if (ba) {
        const rows = ba.samples.map(s => `
            <tr>
                <td class="ba-original">${escapeHtml(s.original)}</td>
                <td class="ba-cleaned">${escapeHtml(s.cleaned)}</td>
            </tr>
        `).join("");
        baBody.innerHTML = `
            <div class="metric-row" style="margin-bottom:16px;">
                <div class="metric-card"><div class="metric-value">${ba.before.vocabulary_size} → ${ba.after.vocabulary_size}</div><div class="metric-label">Vocabulary Size</div></div>
                <div class="metric-card"><div class="metric-value">${ba.before.avg_word_count} → ${ba.after.avg_word_count}</div><div class="metric-label">Avg Words / Doc</div></div>
            </div>
            <table class="ba-table">
                <thead><tr><th>Original</th><th>Cleaned</th></tr></thead>
                <tbody>${rows}</tbody>
            </table>
        `;
        baPanel.style.display = "block";
    } else {
        baPanel.style.display = "none";
    }

    const plotsGrid = document.getElementById("nlp-plots-grid");
    plotsGrid.innerHTML = "";
    for (const [key, filename] of Object.entries(data.plots)) {
        plotsGrid.innerHTML += plotCard(key, `/nlp/view/${filename}`, `/nlp/download/${filename}`);
    }

    const sentiment = data.sentiment; // undefined when include_sentiment was false
    const summaryPanel = document.getElementById("nlp-sentiment-summary");
    const body = document.getElementById("nlp-sentiment-body");
    summaryPanel.style.display = "block";
    body.innerHTML = "";

    if (sentiment) {
        const pills = Object.entries(sentiment.distribution_pct)
            .map(([label, pct]) => `<span class="sentiment-pill ${label}">${label}: ${pct}%</span>`).join("");
        body.innerHTML = `
            <p style="color:var(--txt-dim); font-size:13px; margin-bottom:6px;">
                Rule-based lexicon sentiment — dominant: <b style="color:var(--neon-c);">${sentiment.dominant_sentiment}</b>
            </p>
            <div class="sentiment-pill-row">${pills}</div>
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

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str ?? "";
    return div.innerHTML;
}

document.getElementById("nlp-export-btn")?.addEventListener("click", async () => {
    const textColumn = document.getElementById("nlp-text-column").value;
    if (!textColumn) {
        showToast("Choose a text column first.", "error");
        return;
    }
    const formData = new FormData();
    formData.append("text_column", textColumn);

    try {
        const res = await fetch("/nlp/export", { method: "POST", body: formData });
        if (!res.ok) {
            const data = await res.json().catch(() => ({}));
            showToast(data.message || "Export failed.", "error");
            return;
        }
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "processed_dataset.csv";
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
        showToast("Cleaned dataset exported.", "success");
    } catch (e) {
        showToast("Export failed: " + e.message, "error");
    }
});

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