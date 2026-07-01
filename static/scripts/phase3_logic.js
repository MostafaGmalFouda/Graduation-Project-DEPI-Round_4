/* =====================================================================
   BRight AI — Phase 3 Logic (NLP & RAG)
   Handles: file upload, text column selection, cleaning, analysis,
            vectorization, RAG index building, and the chat interface.
===================================================================== */

const P3_STATE = {
    mode: (typeof P3_MODE !== 'undefined') ? P3_MODE : 'user',
    textColumn: null,
    cleanDone: false,
    ragReady: false,
};

// ── Step navigation ──────────────────────────────────────────────────────
const STEP_ORDER = ['upload', 'clean', 'analyze', 'vectorize', 'rag'];

function unlockStep(step) {
    const btn = document.querySelector(`.step-btn[data-step="${step}"]`);
    if (btn) btn.classList.remove('locked');
}

function goToStep(step) {
    document.querySelectorAll('.step-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.step-btn').forEach(b => b.classList.remove('active'));

    document.getElementById('step-' + step).classList.add('active');
    const btn = document.querySelector(`.step-btn[data-step="${step}"]`);
    if (btn) btn.classList.add('active');
}

document.querySelectorAll('.step-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        if (btn.classList.contains('locked')) return;
        goToStep(btn.dataset.step);
    });
});

// ── Mode-aware visibility ────────────────────────────────────────────────
function applyP3ModeUI() {
    const isDev = P3_STATE.mode === 'developer';
    document.getElementById('mode-title').textContent =
        isDev ? 'Developer Mode — Full Control' : 'User Mode — Guided Flow';

    document.getElementById('p3-dev-steps').style.display = isDev ? 'block' : 'none';
    document.getElementById('p3-dev-analyze-options').style.display = isDev ? 'block' : 'none';
    document.getElementById('p3-dev-vectorize-options').style.display = isDev ? 'block' : 'none';
    document.getElementById('p3-dev-rag-options').style.display = isDev ? 'block' : 'none';
}

// ── Toast helper ─────────────────────────────────────────────────────────
function showP3Toast(message, type = 'success') {
    const existing = document.querySelector('.p3-toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = `p3-toast ${type}`;
    const icon = type === 'error' ? 'fa-circle-exclamation' : 'fa-circle-check';
    toast.innerHTML = `<i class="fas ${icon}"></i> <span>${message}</span>`;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// ── STEP 1: Upload + column detection ───────────────────────────────────
function initP3UploadZone() {
    const zone = document.getElementById('p3-upload-zone');
    const input = document.getElementById('p3-file-input');
    const label = document.getElementById('p3-file-name-label');

    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) {
            input.files = e.dataTransfer.files;
            handleP3FileSelected(e.dataTransfer.files[0]);
        }
    });
    input.addEventListener('change', () => {
        if (input.files.length) handleP3FileSelected(input.files[0]);
    });
}

async function handleP3FileSelected(file) {
    const label = document.getElementById('p3-file-name-label');
    label.textContent = file.name;
    label.style.display = 'inline-block';
    await detectP3Columns(file);
}

async function tryLoadP3FromSession() {
    // Attempt to reuse Phase 1's clean data without requiring a fresh upload
    await detectP3Columns(null);
}

async function detectP3Columns(file) {
    const formData = new FormData();
    if (file) formData.append('file', file);

    try {
        const res = await fetch('/phase3/detect-columns', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.status !== 'success') {
            if (file) showP3Toast(data.message || 'Could not read file.', 'error');
            return;
        }

        document.getElementById('p3-rows-cols').textContent = `${data.rows} rows × ${data.cols} cols`;
        renderP3ColumnPicker(data.text_columns.length ? data.text_columns : data.columns);
        document.getElementById('p3-column-section').style.display = 'block';

        // Data is loaded — stop asking for a file. Hide the upload zone
        // and show a clear confirmation instead, with a way to swap files.
        const zone = document.getElementById('p3-upload-zone');
        zone.style.display = 'none';

        let loadedBanner = document.getElementById('p3-loaded-banner');
        if (!loadedBanner) {
            loadedBanner = document.createElement('div');
            loadedBanner.id = 'p3-loaded-banner';
            loadedBanner.style.cssText = 'max-width:560px;margin:0 auto 22px;padding:12px 20px;border-radius:12px;background:rgba(179,136,255,0.06);border:1px solid var(--p3-panel-border);font-size:12px;color:var(--txt-dim);';
            zone.insertAdjacentElement('afterend', loadedBanner);
        }
        loadedBanner.style.display = 'block';
        loadedBanner.innerHTML = `
            <i class="fas fa-circle-check" style="color:var(--p3-green);"></i>
            <span>${file ? `Loaded "${file.name}"` : 'Using your data from Phase 1'} — ${data.rows} rows × ${data.cols} columns</span>
            <a id="p3-swap-file-link" style="margin-left:10px;color:var(--p3-accent);text-decoration:underline;cursor:pointer;">Use a different file</a>
        `;
        document.getElementById('p3-swap-file-link').onclick = () => {
            zone.style.display = 'block';
            loadedBanner.style.display = 'none';
            document.getElementById('p3-column-section').style.display = 'none';
        };

    } catch (e) {
        if (file) showP3Toast('Connection failed while detecting columns.', 'error');
    }
}

function renderP3ColumnPicker(columns) {
    const wrap = document.getElementById('p3-text-column-picker');
    wrap.innerHTML = '';
    columns.forEach(col => {
        const chip = document.createElement('div');
        chip.className = 'col-pick-chip';
        chip.textContent = col;
        chip.onclick = () => {
            wrap.querySelectorAll('.col-pick-chip').forEach(c => c.classList.remove('selected'));
            chip.classList.add('selected');
            P3_STATE.textColumn = col;
            document.getElementById('p3-confirm-column-btn').disabled = false;
        };
        wrap.appendChild(chip);
    });
}

document.getElementById('p3-confirm-column-btn').onclick = () => {
    if (!P3_STATE.textColumn) return;
    unlockStep('clean');
    goToStep('clean');
};

// ── STEP 2: Clean text ───────────────────────────────────────────────────
document.getElementById('p3-clean-btn').onclick = async () => {
    const btn = document.getElementById('p3-clean-btn');
    const resultsArea = document.getElementById('p3-clean-results');

    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Cleaning...';
    resultsArea.innerHTML = '<div class="p3-loading"><i class="fas fa-spinner fa-spin"></i> Running cleaning pipeline...</div>';

    let steps = 'all';
    if (P3_STATE.mode === 'developer') {
        const checked = Array.from(document.querySelectorAll('#p3-dev-steps input:checked')).map(c => c.value);
        steps = checked.length ? checked.join(',') : 'all';
    }

    try {
        const formData = new FormData();
        formData.append('text_column', P3_STATE.textColumn);
        formData.append('steps', steps);

        const res = await fetch('/phase3/clean-text', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.status === 'success') {
            P3_STATE.cleanDone = true;
            resultsArea.innerHTML = `
                <div class="p3-result-card">
                    <div class="result-title"><i class="fas fa-check-circle" style="color:var(--p3-green);"></i> Cleaning Complete — ${data.rows} rows processed</div>
                    <div class="table-scroll" style="overflow-x:auto;">${data.preview}</div>
                </div>`;
            unlockStep('analyze');
            showP3Toast('Text cleaned successfully.');
        } else {
            resultsArea.innerHTML = `<p style="color:var(--p3-red);">⚠️ ${data.message}</p>`;
            showP3Toast(data.message, 'error');
        }
    } catch (e) {
        resultsArea.innerHTML = `<p style="color:var(--p3-red);">❌ Connection failed.</p>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-bolt"></i> Run Cleaning';
    }
};

// ── STEP 3: Analyze ──────────────────────────────────────────────────────
let p3NgramN = 2;
document.querySelectorAll('#p3-ngram-picker .col-pick-chip').forEach(chip => {
    chip.onclick = () => {
        document.querySelectorAll('#p3-ngram-picker .col-pick-chip').forEach(c => c.classList.remove('selected'));
        chip.classList.add('selected');
        p3NgramN = parseInt(chip.dataset.n, 10);
    };
});

document.getElementById('p3-analyze-btn').onclick = async () => {
    const btn = document.getElementById('p3-analyze-btn');
    const resultsArea = document.getElementById('p3-analyze-results');

    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    resultsArea.innerHTML = '<div class="p3-loading"><i class="fas fa-spinner fa-spin"></i> Computing sentiment, entities, and n-grams...</div>';

    const runNer = P3_STATE.mode === 'developer'
        ? document.getElementById('p3-run-ner').checked
        : true;

    try {
        const formData = new FormData();
        formData.append('ngram_n', String(p3NgramN));
        formData.append('run_ner', runNer ? 'true' : 'false');

        const res = await fetch('/phase3/analyze', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.status === 'success') {
            renderP3AnalysisResults(data);
            unlockStep('vectorize');
            showP3Toast('Analysis complete.');
        } else {
            resultsArea.innerHTML = `<p style="color:var(--p3-red);">⚠️ ${data.message}</p>`;
            showP3Toast(data.message, 'error');
        }
    } catch (e) {
        resultsArea.innerHTML = `<p style="color:var(--p3-red);">❌ Connection failed.</p>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-bolt"></i> Run Analysis';
    }
};

function renderP3AnalysisResults(data) {
    const resultsArea = document.getElementById('p3-analyze-results');
    const s = data.corpus_summary;
    const sentiment = data.sentiment_distribution || {};
    const total = Object.values(sentiment).reduce((a, b) => a + b, 0) || 1;

    let sentimentBars = '';
    ['positive', 'neutral', 'negative'].forEach(label => {
        const count = sentiment[label] || 0;
        const pct = Math.round((count / total) * 100);
        sentimentBars += `
            <div class="sentiment-bar-row">
                <div class="sentiment-bar-label">${label}</div>
                <div class="sentiment-bar-track"><div class="sentiment-bar-fill ${label}" style="width:${pct}%;"></div></div>
                <div class="sentiment-bar-count">${count}</div>
            </div>`;
    });

    let ngramPills = (data.top_ngrams || []).map(g =>
        `<span class="ngram-pill">${g.ngram}<span class="count-badge">×${g.count}</span></span>`
    ).join('');

    let entityHtml = '';
    if (data.entities && data.entities.error) {
        entityHtml = `<p style="font-size:11px;color:var(--txt-dim);">⚠️ NER unavailable: ${data.entities.error}</p>`;
    } else if (data.entities && data.entities.top_entities) {
        entityHtml = data.entities.top_entities.map(([text, count]) =>
            `<span class="entity-pill">${text}<span class="count-badge">×${count}</span></span>`
        ).join('');
    }

    resultsArea.innerHTML = `
        <div class="p3-result-card">
            <div class="result-title"><i class="fas fa-ruler"></i> Corpus Summary</div>
            <div class="stat-grid">
                <div class="stat-tile"><div class="stat-value">${s.total_documents}</div><div class="stat-label">Documents</div></div>
                <div class="stat-tile"><div class="stat-value">${s.vocabulary_size}</div><div class="stat-label">Vocabulary</div></div>
                <div class="stat-tile"><div class="stat-value">${s.avg_word_count}</div><div class="stat-label">Avg Words</div></div>
                <div class="stat-tile"><div class="stat-value">${data.avg_polarity}</div><div class="stat-label">Avg Polarity</div></div>
                <div class="stat-tile"><div class="stat-value">${data.avg_subjectivity}</div><div class="stat-label">Avg Subjectivity</div></div>
            </div>
        </div>

        <div class="p3-result-card">
            <div class="result-title"><i class="fas fa-face-smile"></i> Sentiment Distribution</div>
            ${sentimentBars}
        </div>

        <div class="p3-result-card">
            <div class="result-title"><i class="fas fa-quote-right"></i> Top Recurring Phrases</div>
            <div class="pill-list">${ngramPills || '<span style="font-size:12px;color:var(--txt-dim);">No n-grams found.</span>'}</div>
        </div>

        <div class="p3-result-card">
            <div class="result-title"><i class="fas fa-tags"></i> Named Entities</div>
            <div class="pill-list">${entityHtml || '<span style="font-size:12px;color:var(--txt-dim);">No entities found.</span>'}</div>
        </div>
    `;
}

// ── STEP 4: Vectorize ─────────────────────────────────────────────────────
let p3VecMethod = 'tfidf';
let p3MaxFeatures = 300;

document.querySelectorAll('#p3-vec-method-picker .col-pick-chip').forEach(chip => {
    chip.onclick = () => {
        document.querySelectorAll('#p3-vec-method-picker .col-pick-chip').forEach(c => c.classList.remove('selected'));
        chip.classList.add('selected');
        p3VecMethod = chip.dataset.method;
    };
});

const maxFeatSlider = document.getElementById('p3-maxfeat-slider');
if (maxFeatSlider) {
    maxFeatSlider.oninput = () => {
        p3MaxFeatures = parseInt(maxFeatSlider.value, 10);
        document.getElementById('p3-maxfeat-value').textContent = p3MaxFeatures;
    };
}

document.getElementById('p3-vectorize-btn').onclick = async () => {
    const btn = document.getElementById('p3-vectorize-btn');
    const resultsArea = document.getElementById('p3-vectorize-results');

    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Vectorizing...';
    resultsArea.innerHTML = '<div class="p3-loading"><i class="fas fa-spinner fa-spin"></i> Fitting vectorizer...</div>';

    try {
        const formData = new FormData();
        formData.append('method', P3_STATE.mode === 'developer' ? p3VecMethod : 'tfidf');
        formData.append('max_features', String(P3_STATE.mode === 'developer' ? p3MaxFeatures : 300));

        const res = await fetch('/phase3/vectorize', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.status === 'success') {
            const termPills = (data.top_terms || []).map(t =>
                `<span class="ngram-pill">${t.term}<span class="count-badge">${t.score}</span></span>`
            ).join('');

            resultsArea.innerHTML = `
                <div class="p3-result-card">
                    <div class="result-title"><i class="fas fa-vector-square"></i> Vectorization Result (${data.method.toUpperCase()})</div>
                    <div class="stat-grid">
                        <div class="stat-tile"><div class="stat-value">${data.matrix_shape[0]}</div><div class="stat-label">Documents</div></div>
                        <div class="stat-tile"><div class="stat-value">${data.matrix_shape[1]}</div><div class="stat-label">Features</div></div>
                        <div class="stat-tile"><div class="stat-value">${data.vocabulary_size}</div><div class="stat-label">Vocabulary</div></div>
                    </div>
                    ${termPills ? `<p style="font-size:11px;color:var(--txt-dim);margin:18px 0 10px;text-transform:uppercase;letter-spacing:1px;">Top Weighted Terms</p><div class="pill-list">${termPills}</div>` : ''}
                </div>`;
            unlockStep('rag');
            showP3Toast('Vectorization complete.');
        } else {
            resultsArea.innerHTML = `<p style="color:var(--p3-red);">⚠️ ${data.message}</p>`;
            showP3Toast(data.message, 'error');
        }
    } catch (e) {
        resultsArea.innerHTML = `<p style="color:var(--p3-red);">❌ Connection failed.</p>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-bolt"></i> Run Vectorization';
    }
};

// ── STEP 5: RAG — build index + chat ────────────────────────────────────
const p3ChunkSlider = document.getElementById('p3-chunksize-slider');
const p3OverlapSlider = document.getElementById('p3-overlap-slider');
const p3TopkSlider = document.getElementById('p3-topk-slider');

if (p3ChunkSlider) p3ChunkSlider.oninput = () => document.getElementById('p3-chunksize-value').textContent = p3ChunkSlider.value;
if (p3OverlapSlider) p3OverlapSlider.oninput = () => document.getElementById('p3-overlap-value').textContent = p3OverlapSlider.value;
if (p3TopkSlider) p3TopkSlider.oninput = () => document.getElementById('p3-topk-value').textContent = p3TopkSlider.value;

document.getElementById('p3-build-index-btn').onclick = async () => {
    const btn = document.getElementById('p3-build-index-btn');
    const statusBox = document.getElementById('p3-rag-status');
    const statusText = document.getElementById('p3-rag-status-text');

    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Building Index...';
    statusText.textContent = 'Embedding and indexing chunks...';

    try {
        const formData = new FormData();
        formData.append('chunk_size', String(P3_STATE.mode === 'developer' && p3ChunkSlider ? p3ChunkSlider.value : 120));
        formData.append('chunk_overlap', String(P3_STATE.mode === 'developer' && p3OverlapSlider ? p3OverlapSlider.value : 20));
        formData.append('split_mode', 'token');

        const res = await fetch('/phase3/rag/build-index', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.status === 'success') {
            P3_STATE.ragReady = true;
            statusBox.classList.add('ready');
            statusText.textContent = `Index ready — ${data.chunk_count} chunks, ${data.embedding_dim}-dim embeddings`;
            document.getElementById('p3-chat-window').style.display = 'block';
            showP3Toast('RAG index built successfully.');
        } else {
            statusText.textContent = 'Index build failed.';
            showP3Toast(data.message, 'error');
        }
    } catch (e) {
        statusText.textContent = 'Connection failed.';
        showP3Toast('Connection failed while building index.', 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-database"></i> Rebuild RAG Index';
    }
};

function appendChatBubble(role, html) {
    const messages = document.getElementById('p3-chat-messages');
    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${role}`;
    bubble.innerHTML = html;
    messages.appendChild(bubble);
    messages.scrollTop = messages.scrollHeight;
    return bubble;
}

async function sendP3ChatMessage() {
    const input = document.getElementById('p3-chat-input');
    const question = input.value.trim();
    if (!question || !P3_STATE.ragReady) return;

    appendChatBubble('user', question);
    input.value = '';

    const thinkingBubble = appendChatBubble('assistant', '<div class="typing-dots"><span></span><span></span><span></span></div>');

    try {
        const formData = new FormData();
        formData.append('question', question);
        formData.append('k', String(P3_STATE.mode === 'developer' && p3TopkSlider ? p3TopkSlider.value : 5));

        const res = await fetch('/phase3/rag/ask', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.status === 'success') {
            const sourcesHtml = (data.sources || []).map(s =>
                `<div class="source-item">${s.text}... <span style="opacity:0.5;">(distance ${s.distance})</span></div>`
            ).join('');

            thinkingBubble.innerHTML = `
                ${data.answer}
                ${sourcesHtml ? `<div class="sources-toggle" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display==='flex' ? 'none' : 'flex';">
                    <i class="fas fa-link"></i> ${data.sources.length} source(s)
                </div><div class="sources-list">${sourcesHtml}</div>` : ''}
            `;
        } else {
            thinkingBubble.innerHTML = `⚠️ ${data.message}`;
        }
    } catch (e) {
        thinkingBubble.innerHTML = '❌ Connection failed.';
    }
}

document.getElementById('p3-chat-send-btn').onclick = sendP3ChatMessage;
document.getElementById('p3-chat-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendP3ChatMessage();
});

// ── Init ──────────────────────────────────────────────────────────────────
window.onload = () => {
    applyP3ModeUI();
    initP3UploadZone();
    tryLoadP3FromSession();
};
