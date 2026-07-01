/* =====================================================================
   BRight AI — Unified AI Pipeline Logic (ML -> NLP -> RAG)
   One connected journey. Reuses the existing /phase4/* (ML) and
   /phase3/* (NLP + RAG) JSON APIs under the hood — this file is purely
   the orchestration/UI layer that chains them together.
===================================================================== */

const AI_STATE = {
    mode: (typeof AI_MODE !== 'undefined') ? AI_MODE : 'user',
    // shared dataset info
    columns: [],
    target: null,
    textColumn: null,
    // ML
    taskType: null,
    selectedModel: 'random_forest',
    trainResult: null,
    availableModels: { classification: [], regression: [], clustering: [] },
    // NLP
    cleanDone: false,
    // RAG
    ragReady: false,
};

/* ════════════════════════════════════════════
   MASTER JOURNEY TRACKER (ML -> NLP -> RAG)
════════════════════════════════════════════ */
function setJourneyStage(stage, status) {
    // status: 'locked' | 'unlocked' | 'current' | 'done'
    const el = document.querySelector(`.journey-stage[data-stage="${stage}"]`);
    if (!el) return;
    el.classList.remove('locked', 'unlocked', 'current', 'done');
    el.classList.add(status);
}

function markConnectorDone(afterStage) {
    const stages = ['ml', 'nlp', 'rag'];
    const idx = stages.indexOf(afterStage);
    const connectors = document.querySelectorAll('.journey-connector');
    if (connectors[idx]) connectors[idx].classList.add('done');
}

function goToMajorStage(stage) {
    document.querySelectorAll('.major-panel').forEach(p => p.classList.remove('active'));
    document.getElementById('major-' + stage).classList.add('active');

    ['ml', 'nlp', 'rag'].forEach(s => {
        const el = document.querySelector(`.journey-stage[data-stage="${s}"]`);
        if (!el) return;
        if (s === stage) {
            el.classList.remove('locked', 'done');
            el.classList.add('current', 'unlocked', s + '-stage');
        } else if (el.classList.contains('done')) {
            // leave as done
        } else if (!el.classList.contains('unlocked')) {
            el.classList.add('locked');
        } else {
            el.classList.remove('current');
        }
    });
}

document.querySelectorAll('.journey-stage').forEach(el => {
    el.addEventListener('click', () => {
        if (el.classList.contains('locked')) return;
        cancelPendingAutoContinue();
        goToMajorStage(el.dataset.stage);
    });
});

/* ════════════════════════════════════════════
   SUB-STEP NAVIGATION (within a major stage)
════════════════════════════════════════════ */
function unlockSub(subId) {
    const btn = document.querySelector(`.substep-btn[data-sub="${subId}"]`);
    if (btn) btn.classList.remove('locked');
}

function goToSub(subId) {
    const panel = document.getElementById('sub-' + subId);
    if (!panel) return;
    const parent = panel.closest('.major-panel');

    parent.querySelectorAll('.sub-panel').forEach(p => p.classList.remove('active'));
    parent.querySelectorAll('.substep-btn').forEach(b => b.classList.remove('active'));

    panel.classList.add('active');
    const btn = parent.querySelector(`.substep-btn[data-sub="${subId}"]`);
    if (btn) btn.classList.add('active');
}

document.querySelectorAll('.substep-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        if (btn.classList.contains('locked')) return;
        cancelPendingAutoContinue();
        goToSub(btn.dataset.sub);
    });
});

/* ════════════════════════════════════════════
   MODE-AWARE UI
════════════════════════════════════════════ */
function applyAiModeUI() {
    const isDev = AI_STATE.mode === 'developer';
    document.getElementById('ai-mode-title').textContent =
        isDev ? 'Developer Mode — Full Control Over Every Stage' : 'User Mode — Fully Automated Journey';

    document.getElementById('ai-user-train-block').style.display = isDev ? 'none' : 'block';
    document.getElementById('ai-dev-train-block').style.display = isDev ? 'block' : 'none';
    document.getElementById('ai-dev-clean-steps').style.display = isDev ? 'block' : 'none';
    document.getElementById('ai-dev-analyze-options').style.display = isDev ? 'block' : 'none';
    document.getElementById('ai-dev-vectorize-options').style.display = isDev ? 'block' : 'none';
    document.getElementById('ai-dev-rag-options').style.display = isDev ? 'block' : 'none';

    // Developer Mode keeps the manual "continue" buttons between major
    // stages visible immediately so technical users can jump ahead
    // without finishing every sub-step if they choose to.
    if (isDev) {
        document.getElementById('ai-ml-done-banner').style.display = 'block';
        document.getElementById('ai-nlp-done-banner').style.display = 'block';
    }
}

function showAiToast(message, type = 'success') {
    const existing = document.querySelector('.ai-toast');
    if (existing) existing.remove();
    const toast = document.createElement('div');
    toast.className = `ai-toast ${type}`;
    const icon = type === 'error' ? 'fa-circle-exclamation' : 'fa-circle-check';
    toast.innerHTML = `<i class="fas ${icon}"></i> <span>${message}</span>`;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// ── Auto-continue countdown helper (User Mode connective tissue) ───────
// Every automatic transition is fully cancelable: "Skip now" jumps ahead
// immediately, and "Stay here" cancels the countdown entirely so the
// person can keep reviewing the current step for as long as they like.
// Only one auto-continue banner is ever active at a time — starting a
// new one clears any previous pending countdown first.
let _activeAutoContinueInterval = null;

function autoContinue(containerEl, label, destinationFn, seconds = 4) {
    // Cancel any previous countdown still running elsewhere on the page
    if (_activeAutoContinueInterval) {
        clearInterval(_activeAutoContinueInterval);
        _activeAutoContinueInterval = null;
    }
    document.querySelectorAll('.auto-continue-banner').forEach(b => b.remove());

    const banner = document.createElement('div');
    banner.className = 'auto-continue-banner';
    banner.innerHTML = `
        <i class="fas fa-bolt" style="color:var(--ai-green);"></i>
        ${label} in <span class="count">${seconds}</span>s
        <a id="ai-skip-auto-${Date.now()}" class="ai-auto-skip-link">Skip now</a>
        <a id="ai-stay-auto-${Date.now()}" class="ai-auto-stay-link">Stay on this step</a>
    `;
    containerEl.appendChild(banner);

    let secondsLeft = seconds;
    const countEl = banner.querySelector('.count');
    const interval = setInterval(() => {
        secondsLeft -= 1;
        if (countEl) countEl.innerText = secondsLeft;
        if (secondsLeft <= 0) {
            clearInterval(interval);
            _activeAutoContinueInterval = null;
            banner.remove();
            destinationFn();
        }
    }, 1000);
    _activeAutoContinueInterval = interval;

    banner.querySelector('.ai-auto-skip-link').onclick = () => {
        clearInterval(interval);
        _activeAutoContinueInterval = null;
        banner.remove();
        destinationFn();
    };

    banner.querySelector('.ai-auto-stay-link').onclick = () => {
        clearInterval(interval);
        _activeAutoContinueInterval = null;
        banner.remove();
        showAiToast('Staying here — continue whenever you\'re ready.');
    };
}

// Manually navigating via the journey tracker or sub-step nav should
// always cancel any pending auto-continue, so clicking around never
// fights with a countdown that's about to fire underneath the person.
function cancelPendingAutoContinue() {
    if (_activeAutoContinueInterval) {
        clearInterval(_activeAutoContinueInterval);
        _activeAutoContinueInterval = null;
    }
    document.querySelectorAll('.auto-continue-banner').forEach(b => b.remove());
}

/* =====================================================================
   STAGE 1 — MACHINE LEARNING
===================================================================== */

function initAiUploadZone() {
    const zone = document.getElementById('ai-upload-zone');
    const input = document.getElementById('ai-file-input');

    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) {
            input.files = e.dataTransfer.files;
            handleAiFileSelected(e.dataTransfer.files[0]);
        }
    });
    input.addEventListener('change', () => {
        if (input.files.length) handleAiFileSelected(input.files[0]);
    });
}

async function handleAiFileSelected(file) {
    const label = document.getElementById('ai-file-name-label');
    label.textContent = file.name;
    label.style.display = 'inline-block';
    await detectAiColumns(file);
}

async function tryLoadAiFromSession() {
    await detectAiColumns(null);
}

async function detectAiColumns(file) {
    const formData = new FormData();
    if (file) formData.append('file', file);

    try {
        const res = await fetch('/phase4/detect-columns', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.status !== 'success') {
            if (file) showAiToast(data.message || 'Could not read file.', 'error');
            return;
        }

        document.getElementById('ai-rows-cols').textContent = `${data.rows} rows × ${data.cols} cols`;
        AI_STATE.columns = data.columns;
        renderAiTargetPicker(data.columns);
        document.getElementById('ai-target-section').style.display = 'block';

        // Once we have a dataset — whether freshly uploaded just now, or
        // already sitting in the session from Phase 1 / Phase 2 — there's
        // no reason to keep asking for a file. Hide the upload zone and
        // show a clear "loaded" state instead, with a small affordance to
        // swap the dataset if the person genuinely wants to.
        const zone = document.getElementById('ai-upload-zone');
        zone.style.display = 'none';

        const banner = document.getElementById('ai-session-banner');
        banner.style.display = 'block';
        banner.innerHTML = `
            <i class="fas fa-circle-check" style="color:var(--ai-green);"></i>
            <span>${file ? `Loaded "${file.name}"` : 'Using your data from Phase 1 / Phase 2'} — ${data.rows} rows × ${data.cols} columns</span>
            <a id="ai-swap-file-link" style="margin-left:10px;color:var(--ai-accent);text-decoration:underline;cursor:pointer;">Use a different file</a>
        `;
        const swapLink = document.getElementById('ai-swap-file-link');
        if (swapLink) {
            swapLink.onclick = () => {
                zone.style.display = 'block';
                banner.style.display = 'none';
                document.getElementById('ai-target-section').style.display = 'none';
            };
        }

    } catch (e) {
        if (file) showAiToast('Connection failed while detecting columns.', 'error');
    }
}

function renderAiTargetPicker(columns) {
    const wrap = document.getElementById('ai-target-picker');
    wrap.innerHTML = '';
    columns.forEach(col => {
        const chip = document.createElement('div');
        chip.className = 'pick-chip';
        chip.innerHTML = `${col.name} <span style="opacity:0.5;font-size:9px;">${col.dtype}</span>`;
        chip.onclick = () => {
            wrap.querySelectorAll('.pick-chip').forEach(c => c.classList.remove('selected', 'ml-sel'));
            chip.classList.add('selected', 'ml-sel');
            AI_STATE.target = col.name;
            document.getElementById('ai-confirm-target-btn').disabled = false;
        };
        wrap.appendChild(chip);
    });
}

document.getElementById('ai-skip-target').onclick = () => {
    AI_STATE.target = null;
    document.querySelectorAll('#ai-target-picker .pick-chip').forEach(c => c.classList.remove('selected', 'ml-sel'));
    document.getElementById('ai-confirm-target-btn').disabled = false;
    showAiToast('Clustering mode — no target selected.');
};

document.getElementById('ai-confirm-target-btn').onclick = async () => {
    unlockSub('ml-train');
    goToSub('ml-train');

    if (AI_STATE.mode === 'developer') {
        await loadAiAvailableModels();
    } else {
        await loadAiRecommendations();
    }
};

// ── ML: User Mode recommendations ────────────────────────────────────
async function loadAiRecommendations() {
    const grid = document.getElementById('ai-recommend-grid');
    grid.innerHTML = '<div class="ai-loading"><i class="fas fa-spinner fa-spin"></i> Getting recommendations...</div>';

    try {
        const formData = new FormData();
        if (AI_STATE.target) formData.append('target', AI_STATE.target);

        const res = await fetch('/phase4/recommend', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.status === 'success') {
            AI_STATE.taskType = data.task_type;
            grid.innerHTML = data.recommendations.map((r, i) => `
                <div class="recommend-card ${i === 0 ? 'top-pick' : ''}">
                    ${i === 0 ? '<div class="rank-badge">TOP PICK</div>' : ''}
                    <div class="model-name"><i class="fas fa-cube"></i> ${r.model_name.replace(/_/g, ' ')}</div>
                    <div class="model-reason">${r.reason}</div>
                </div>
            `).join('');

            // User Mode: train automatically right after recommendations land
            autoContinue(grid.parentElement, 'Auto-training the top recommended model', () => runAiTrain('user'), 3);
        } else {
            grid.innerHTML = `<p style="color:var(--ai-ml);">⚠️ ${data.message}</p>`;
        }
    } catch (e) {
        grid.innerHTML = `<p style="color:var(--ai-ml);">❌ Connection failed.</p>`;
    }
}

document.getElementById('ai-user-train-btn').onclick = async () => { await runAiTrain('user'); };

// ── ML: Developer Mode manual model + params ─────────────────────────
// CRITICAL FIX: the model grid must only ever show models that are
// actually compatible with the detected task_type. Previously this
// merged classification + regression + clustering model names into a
// single undifferentiated list, so picking e.g. "knn" on a continuous
// (regression) target silently built a KNeighborsClassifier and crashed
// training with "Unknown label type: continuous". We now resolve the
// real task_type FIRST (same inference the backend uses), then only
// ever offer models that support it.
async function loadAiAvailableModels() {
    try {
        const [modelsRes, taskRes] = await Promise.all([
            fetch('/phase4/available-models'),
            fetch('/phase4/recommend', {
                method: 'POST',
                body: (() => { const fd = new FormData(); if (AI_STATE.target) fd.append('target', AI_STATE.target); return fd; })(),
            }),
        ]);

        const modelsData = await modelsRes.json();
        const taskData = await taskRes.json();

        if (modelsData.status === 'success') AI_STATE.availableModels = modelsData.models;
        AI_STATE.taskType = (taskData.status === 'success') ? taskData.task_type : 'classification';

        // pick a sensible default model that's actually valid for this task
        const compatible = AI_STATE.availableModels[AI_STATE.taskType] || [];
        if (!compatible.includes(AI_STATE.selectedModel)) {
            AI_STATE.selectedModel = compatible[0] || AI_STATE.selectedModel;
        }

        renderAiModelGrid();
    } catch (e) {
        showAiToast('Could not load model list.', 'error');
    }
}

function renderAiModelGrid() {
    const grid = document.getElementById('ai-model-select-grid');
    const compatible = AI_STATE.availableModels[AI_STATE.taskType] || [];

    // Show the detected task type so Developer Mode users understand why
    // the list is filtered the way it is.
    let taskBanner = document.getElementById('ai-task-type-banner');
    if (!taskBanner) {
        taskBanner = document.createElement('p');
        taskBanner.id = 'ai-task-type-banner';
        taskBanner.style.cssText = 'font-size:11px;color:var(--txt-dim);margin-bottom:12px;';
        grid.parentElement.insertBefore(taskBanner, grid);
    }
    taskBanner.innerHTML = `<i class="fas fa-bullseye" style="color:var(--ai-ml);"></i> Detected task: <strong style="color:var(--ai-ml);">${AI_STATE.taskType}</strong> — only compatible models are shown.`;

    grid.innerHTML = '';
    compatible.forEach(name => {
        const opt = document.createElement('div');
        opt.className = 'model-select-option' + (name === AI_STATE.selectedModel ? ' selected' : '');
        opt.textContent = name.replace(/_/g, ' ');
        opt.onclick = () => {
            grid.querySelectorAll('.model-select-option').forEach(o => o.classList.remove('selected'));
            opt.classList.add('selected');
            AI_STATE.selectedModel = name;
            renderAiParamFields(name);
        };
        grid.appendChild(opt);
    });
    renderAiParamFields(AI_STATE.selectedModel);
}

const AI_MODEL_PARAM_HINTS = {
    random_forest: [{ key: 'n_estimators', label: 'n_estimators', placeholder: '100' }, { key: 'max_depth', label: 'max_depth (blank = none)', placeholder: '' }],
    logistic_regression: [{ key: 'C', label: 'C (inverse regularization)', placeholder: '1.0' }, { key: 'max_iter', label: 'max_iter', placeholder: '1000' }],
    linear_regression: [],
    svm: [{ key: 'C', label: 'C', placeholder: '1.0' }, { key: 'kernel', label: 'kernel (rbf/linear/poly)', placeholder: 'rbf' }],
    knn: [{ key: 'n_neighbors', label: 'n_neighbors', placeholder: '5' }],
    decision_tree: [{ key: 'max_depth', label: 'max_depth (blank = none)', placeholder: '' }],
    kmeans: [{ key: 'n_clusters', label: 'n_clusters', placeholder: '3' }],
    dbscan: [{ key: 'eps', label: 'eps', placeholder: '0.5' }, { key: 'min_samples', label: 'min_samples', placeholder: '5' }],
};

function renderAiParamFields(modelName) {
    const container = document.getElementById('ai-params-fields');
    const fields = AI_MODEL_PARAM_HINTS[modelName] || [];
    if (!fields.length) {
        container.innerHTML = '<p style="font-size:11px;color:var(--txt-dim);">This model uses default parameters.</p>';
        return;
    }
    container.innerHTML = fields.map(f => `
        <div class="param-field">
            <label>${f.label}</label>
            <input type="text" data-param="${f.key}" placeholder="${f.placeholder}">
        </div>
    `).join('');
}

document.getElementById('ai-tuning-switch').onclick = function () {
    const isOn = this.classList.toggle('on');
    document.getElementById('ai-tuning-options').style.display = isOn ? 'block' : 'none';
};

document.querySelectorAll('[data-tune]').forEach(opt => {
    opt.onclick = () => {
        document.querySelectorAll('[data-tune]').forEach(o => o.classList.remove('selected'));
        opt.classList.add('selected');
    };
});

const aiCvSlider = document.getElementById('ai-cv-slider');
if (aiCvSlider) aiCvSlider.oninput = () => document.getElementById('ai-cv-value').textContent = aiCvSlider.value;

const aiTestSizeSlider = document.getElementById('ai-testsize-slider');
if (aiTestSizeSlider) {
    aiTestSizeSlider.oninput = () => {
        document.getElementById('ai-testsize-value').textContent = Math.round(parseFloat(aiTestSizeSlider.value) * 100) + '%';
    };
}

document.getElementById('ai-dev-train-btn').onclick = async () => { await runAiTrain('developer'); };

async function runAiTrain(mode) {
    const statusBox = document.getElementById('ai-training-status');
    statusBox.style.display = 'flex';

    const trainBtn = mode === 'user'
        ? document.getElementById('ai-user-train-btn')
        : document.getElementById('ai-dev-train-btn');
    if (trainBtn) trainBtn.disabled = true;

    try {
        const formData = new FormData();
        formData.append('run_mode', mode);
        if (AI_STATE.target) formData.append('target', AI_STATE.target);

        if (mode === 'developer') {
            formData.append('model_name', AI_STATE.selectedModel);

            const params = {};
            document.querySelectorAll('#ai-params-fields input[data-param]').forEach(input => {
                if (input.value.trim() !== '') {
                    const num = Number(input.value);
                    params[input.dataset.param] = isNaN(num) ? input.value.trim() : num;
                }
            });
            formData.append('params', JSON.stringify(params));

            const tuningOn = document.getElementById('ai-tuning-switch').classList.contains('on');
            if (tuningOn) {
                const method = document.querySelector('[data-tune].selected')?.dataset.tune || 'grid';
                const paramSpace = buildAiDefaultParamSpace(AI_STATE.selectedModel);
                formData.append('tuning', JSON.stringify({ method, param_space: paramSpace }));
            }

            formData.append('cv', aiCvSlider ? aiCvSlider.value : '5');
            formData.append('test_size', aiTestSizeSlider ? aiTestSizeSlider.value : '0.2');
        }

        const res = await fetch('/phase4/train', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.status === 'success') {
            AI_STATE.trainResult = data.result;
            AI_STATE.taskType = data.result.task_type;
            renderAiResults(data.result);
            unlockSub('ml-results');
            unlockSub('ml-predict');
            renderAiPredictForm();
            goToSub('ml-results');
            showAiToast('Model trained successfully!');

            if (AI_STATE.mode === 'user') {
                document.getElementById('ai-ml-done-banner').style.display = 'block';
            }
        } else {
            showAiToast(data.message || 'Training failed.', 'error');
        }
    } catch (e) {
        showAiToast('Connection failed during training.', 'error');
    } finally {
        statusBox.style.display = 'none';
        if (trainBtn) trainBtn.disabled = false;
    }
}

function buildAiDefaultParamSpace(modelName) {
    const spaces = {
        random_forest: { n_estimators: [50, 100, 200], max_depth: [5, 10, null] },
        logistic_regression: { C: [0.1, 1.0, 10.0] },
        svm: { C: [0.1, 1.0, 10.0] },
        knn: { n_neighbors: [3, 5, 7, 9] },
        decision_tree: { max_depth: [3, 5, 10, null] },
    };
    return spaces[modelName] || {};
}

function renderAiResults(result) {
    const summaryBar = document.getElementById('ai-model-summary');
    const metricsGrid = document.getElementById('ai-metrics-grid');
    const tuningBox = document.getElementById('ai-tuning-summary');

    const modelLabel = (result.chosen_model || result.model_name || '—').replace(/_/g, ' ');
    summaryBar.innerHTML = `
        <span><i class="fas fa-cube"></i> Model: <span class="chip">${modelLabel}</span></span>
        <span><i class="fas fa-bullseye"></i> Task: <span class="chip">${result.task_type}</span></span>
        <span><i class="fas fa-toggle-on"></i> Mode: <span class="chip">${result.mode}</span></span>
    `;

    const m = result.metrics || {};
    let tiles = '';
    if (result.task_type === 'classification') {
        tiles = `
            ${aiMetricTile(m.accuracy, 'Accuracy')}
            ${aiMetricTile(m.precision, 'Precision')}
            ${aiMetricTile(m.recall, 'Recall')}
            ${aiMetricTile(m.f1_score, 'F1 Score')}
            ${m.roc_auc != null ? aiMetricTile(m.roc_auc, 'ROC-AUC') : ''}
        `;
    } else if (result.task_type === 'regression') {
        tiles = `
            ${aiMetricTile(m.r2, 'R²')}
            ${aiMetricTile(m.mae, 'MAE', false)}
            ${aiMetricTile(m.rmse, 'RMSE', false)}
            ${aiMetricTile(m.mse, 'MSE', false)}
        `;
    } else {
        tiles = `<p style="font-size:12px;color:var(--txt-dim);grid-column:1/-1;">${(m.note || 'Clustering complete.')}</p>`;
    }
    metricsGrid.innerHTML = tiles;
    document.querySelectorAll('#ai-metrics-grid .stat-tile').forEach(t => t.classList.add('ml-tile'));

    if (result.tuning_results) {
        const tr = result.tuning_results;
        tuningBox.style.display = 'block';
        tuningBox.innerHTML = `
            <div class="result-title" style="color:var(--ai-ml);">
                <i class="fas fa-magnifying-glass-chart"></i> Hyperparameter Tuning Result
            </div>
            <p style="font-size:12px;color:var(--txt-dim);">Best Score: <strong style="color:var(--ai-ml);">${tr.best_score?.toFixed ? tr.best_score.toFixed(4) : tr.best_score}</strong></p>
            <p style="font-size:12px;color:var(--txt-dim);margin-top:6px;">Best Params: <code>${JSON.stringify(tr.best_params)}</code></p>
        `;
    } else {
        tuningBox.style.display = 'none';
    }

    renderAiPlotButtons(result.task_type);
}

function aiMetricTile(value, label, isPct = true) {
    const display = (value == null) ? '—' : (isPct ? (value * 100).toFixed(1) + '%' : (value.toFixed ? value.toFixed(3) : value));
    return `<div class="stat-tile"><div class="stat-value">${display}</div><div class="stat-label">${label}</div></div>`;
}

function renderAiPlotButtons(taskType) {
    const buttonsRow = document.getElementById('ai-plot-buttons');
    const plots = taskType === 'classification'
        ? [['confusion_matrix', 'Confusion Matrix'], ['roc_curve', 'ROC Curve'], ['feature_importance', 'Feature Importance']]
        : taskType === 'regression'
        ? [['actual_vs_predicted', 'Actual vs Predicted'], ['feature_importance', 'Feature Importance'], ['correlation_matrix', 'Correlation Matrix']]
        : [];

    buttonsRow.innerHTML = plots.map(([type, label]) =>
        `<button class="plot-pick-btn" data-plot="${type}">${label}</button>`
    ).join('');

    buttonsRow.querySelectorAll('.plot-pick-btn').forEach(btn => {
        btn.onclick = () => generateAiPlot(btn.dataset.plot, btn);
    });
}

async function generateAiPlot(plotType, btnEl) {
    const gallery = document.getElementById('ai-plot-gallery');
    const originalLabel = btnEl ? btnEl.textContent : '';
    if (btnEl) { btnEl.disabled = true; btnEl.innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${originalLabel}`; }

    try {
        const formData = new FormData();
        formData.append('plot_type', plotType);
        const res = await fetch('/phase4/plot', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.status === 'success') {
            const card = document.createElement('div');
            card.className = 'plot-card';
            card.innerHTML = `
                <div class="plot-title">${plotType.replace(/_/g, ' ')}</div>
                <img src="${data.view_url}" onclick="openAiLightbox('${data.view_url}')">
            `;
            gallery.prepend(card);
        } else {
            showAiToast(data.message || 'Could not generate plot.', 'error');
        }
    } catch (e) {
        showAiToast('Connection failed while generating plot.', 'error');
    } finally {
        if (btnEl) { btnEl.disabled = false; btnEl.innerHTML = originalLabel; }
    }
}

function openAiLightbox(url) {
    document.getElementById('ai-lightbox-img').src = url;
    document.getElementById('ai-lightbox').style.display = 'block';
}
function closeAiLightbox() {
    document.getElementById('ai-lightbox').style.display = 'none';
}

document.getElementById('ai-goto-predict-btn').onclick = () => goToSub('ml-predict');

function renderAiPredictForm() {
    const form = document.getElementById('ai-predict-form');
    const featureCols = AI_STATE.columns
        .map(c => c.name)
        .filter(name => name !== AI_STATE.target);

    form.innerHTML = featureCols.map(name => `
        <div class="predict-field">
            <label>${name}</label>
            <input type="text" data-feature="${name}" placeholder="value">
        </div>
    `).join('');
}

document.getElementById('ai-predict-btn').onclick = async () => {
    const resultBox = document.getElementById('ai-prediction-result');
    const sample = {};
    document.querySelectorAll('#ai-predict-form input[data-feature]').forEach(input => {
        const num = Number(input.value);
        sample[input.dataset.feature] = (input.value.trim() !== '' && !isNaN(num)) ? num : input.value;
    });

    resultBox.innerHTML = '<div class="ai-loading"><i class="fas fa-spinner fa-spin"></i> Predicting...</div>';

    try {
        const formData = new FormData();
        formData.append('sample', JSON.stringify(sample));
        const res = await fetch('/phase4/predict', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.status === 'success') {
            const r = data.result;
            let probsHtml = '';
            if (r.probabilities) {
                probsHtml = Object.entries(r.probabilities).map(([cls, p]) =>
                    `<div class="contrib-row">
                        <div class="contrib-label">${cls}</div>
                        <div class="contrib-track"><div class="contrib-fill" style="width:${(p * 100).toFixed(0)}%;"></div></div>
                        <div style="font-size:11px;color:var(--txt-dim);width:50px;">${(p * 100).toFixed(1)}%</div>
                    </div>`
                ).join('');
            }

            let contribHtml = '';
            if (r.top_contributing_features) {
                const maxImp = Math.max(...r.top_contributing_features.map(f => Math.abs(f.importance ?? f.contribution ?? 0)));
                contribHtml = r.top_contributing_features.slice(0, 6).map(f => {
                    const val = f.importance ?? f.contribution ?? 0;
                    const pct = maxImp ? (Math.abs(val) / maxImp) * 100 : 0;
                    return `<div class="contrib-row">
                        <div class="contrib-label">${f.feature}</div>
                        <div class="contrib-track"><div class="contrib-fill" style="width:${pct}%;"></div></div>
                        <div style="font-size:11px;color:var(--txt-dim);width:50px;">${val.toFixed ? val.toFixed(3) : val}</div>
                    </div>`;
                }).join('');
            }

            resultBox.innerHTML = `
                <div class="prediction-result">
                    <div class="pred-value">${r.prediction}</div>
                    <div class="pred-label">Predicted Value</div>
                    ${probsHtml ? `<div style="margin-top:20px;text-align:left;"><p style="font-size:10px;color:var(--txt-dim);text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">Class Probabilities</p>${probsHtml}</div>` : ''}
                    ${contribHtml ? `<div style="margin-top:20px;text-align:left;"><p style="font-size:10px;color:var(--txt-dim);text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">Top Contributing Features</p>${contribHtml}</div>` : ''}
                </div>
            `;

            document.getElementById('ai-ml-done-banner').style.display = 'block';
            if (AI_STATE.mode === 'user') {
                autoContinue(document.getElementById('ai-ml-done-banner'), 'Moving on to NLP — Text Intelligence', goToNlpStage, 4);
            }
        } else {
            resultBox.innerHTML = `<p style="color:var(--ai-ml);">⚠️ ${data.message}</p>`;
        }
    } catch (e) {
        resultBox.innerHTML = `<p style="color:var(--ai-ml);">❌ Connection failed.</p>`;
    }
};

document.getElementById('ai-continue-to-nlp-btn').onclick = () => goToNlpStage();

/* =====================================================================
   STAGE 2 — NLP
===================================================================== */

function goToNlpStage() {
    setJourneyStage('ml', 'done');
    markConnectorDone('ml');
    goToMajorStage('nlp');
    loadNlpTextColumns();
}

async function loadNlpTextColumns() {
    const formData = new FormData();
    try {
        const res = await fetch('/phase3/detect-columns', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.status === 'success') {
            document.getElementById('ai-nlp-rows-cols').textContent = `${data.rows} rows × ${data.cols} cols`;
            renderNlpColumnPicker(data.text_columns.length ? data.text_columns : data.columns);
        }
    } catch (e) {
        showAiToast('Could not load columns for NLP.', 'error');
    }
}

function renderNlpColumnPicker(columns) {
    const wrap = document.getElementById('ai-text-column-picker');
    wrap.innerHTML = '';
    columns.forEach(col => {
        const chip = document.createElement('div');
        chip.className = 'pick-chip';
        chip.textContent = col;
        chip.onclick = () => {
            wrap.querySelectorAll('.pick-chip').forEach(c => c.classList.remove('selected', 'nlp-sel'));
            chip.classList.add('selected', 'nlp-sel');
            AI_STATE.textColumn = col;
            document.getElementById('ai-confirm-textcol-btn').disabled = false;
        };
        wrap.appendChild(chip);
    });
}

document.getElementById('ai-confirm-textcol-btn').onclick = () => {
    if (!AI_STATE.textColumn) return;
    unlockSub('nlp-clean');
    goToSub('nlp-clean');
    if (AI_STATE.mode === 'user') {
        document.getElementById('ai-clean-btn').click();
    }
};

document.getElementById('ai-clean-btn').onclick = async () => {
    const btn = document.getElementById('ai-clean-btn');
    const resultsArea = document.getElementById('ai-clean-results');

    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Cleaning...';
    resultsArea.innerHTML = '<div class="ai-loading"><i class="fas fa-spinner fa-spin"></i> Running cleaning pipeline...</div>';

    let steps = 'all';
    if (AI_STATE.mode === 'developer') {
        const checked = Array.from(document.querySelectorAll('#ai-dev-clean-steps input:checked')).map(c => c.value);
        steps = checked.length ? checked.join(',') : 'all';
    }

    try {
        const formData = new FormData();
        formData.append('text_column', AI_STATE.textColumn);
        formData.append('steps', steps);

        const res = await fetch('/phase3/clean-text', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.status === 'success') {
            AI_STATE.cleanDone = true;
            resultsArea.innerHTML = `
                <div class="result-card">
                    <div class="result-title nlp-text"><i class="fas fa-check-circle" style="color:var(--ai-green);"></i> Cleaning Complete — ${data.rows} rows processed</div>
                    <div style="overflow-x:auto;">${data.preview}</div>
                </div>`;
            unlockSub('nlp-analyze');
            showAiToast('Text cleaned successfully.');

            if (AI_STATE.mode === 'user') {
                autoContinue(resultsArea, 'Continuing to Analysis', () => { goToSub('nlp-analyze'); document.getElementById('ai-analyze-btn').click(); }, 3);
            }
        } else {
            resultsArea.innerHTML = `<p style="color:var(--ai-red);">⚠️ ${data.message}</p>`;
            showAiToast(data.message, 'error');
        }
    } catch (e) {
        resultsArea.innerHTML = `<p style="color:var(--ai-red);">❌ Connection failed.</p>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-bolt"></i> Run Cleaning';
    }
};

let aiNgramN = 2;
document.querySelectorAll('#ai-ngram-picker .pick-chip').forEach(chip => {
    chip.onclick = () => {
        document.querySelectorAll('#ai-ngram-picker .pick-chip').forEach(c => c.classList.remove('selected', 'nlp-sel'));
        chip.classList.add('selected', 'nlp-sel');
        aiNgramN = parseInt(chip.dataset.n, 10);
    };
});

document.getElementById('ai-analyze-btn').onclick = async () => {
    const btn = document.getElementById('ai-analyze-btn');
    const resultsArea = document.getElementById('ai-analyze-results');

    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    resultsArea.innerHTML = '<div class="ai-loading"><i class="fas fa-spinner fa-spin"></i> Computing sentiment, entities, and n-grams...</div>';

    const runNer = AI_STATE.mode === 'developer' ? document.getElementById('ai-run-ner').checked : true;

    try {
        const formData = new FormData();
        formData.append('ngram_n', String(aiNgramN));
        formData.append('run_ner', runNer ? 'true' : 'false');

        const res = await fetch('/phase3/analyze', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.status === 'success') {
            renderNlpAnalysisResults(data);
            unlockSub('nlp-vectorize');
            showAiToast('Analysis complete.');

            if (AI_STATE.mode === 'user') {
                autoContinue(resultsArea, 'Continuing to Vectorization', () => { goToSub('nlp-vectorize'); document.getElementById('ai-vectorize-btn').click(); }, 3);
            }
        } else {
            resultsArea.innerHTML = `<p style="color:var(--ai-red);">⚠️ ${data.message}</p>`;
            showAiToast(data.message, 'error');
        }
    } catch (e) {
        resultsArea.innerHTML = `<p style="color:var(--ai-red);">❌ Connection failed.</p>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-bolt"></i> Run Analysis';
    }
};

function renderNlpAnalysisResults(data) {
    const resultsArea = document.getElementById('ai-analyze-results');
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
        <div class="result-card">
            <div class="result-title nlp-text"><i class="fas fa-ruler"></i> Corpus Summary</div>
            <div class="tile-grid">
                <div class="stat-tile nlp-tile"><div class="stat-value">${s.total_documents}</div><div class="stat-label">Documents</div></div>
                <div class="stat-tile nlp-tile"><div class="stat-value">${s.vocabulary_size}</div><div class="stat-label">Vocabulary</div></div>
                <div class="stat-tile nlp-tile"><div class="stat-value">${s.avg_word_count}</div><div class="stat-label">Avg Words</div></div>
                <div class="stat-tile nlp-tile"><div class="stat-value">${data.avg_polarity}</div><div class="stat-label">Avg Polarity</div></div>
                <div class="stat-tile nlp-tile"><div class="stat-value">${data.avg_subjectivity}</div><div class="stat-label">Avg Subjectivity</div></div>
            </div>
        </div>
        <div class="result-card">
            <div class="result-title nlp-text"><i class="fas fa-face-smile"></i> Sentiment Distribution</div>
            ${sentimentBars}
        </div>
        <div class="result-card">
            <div class="result-title nlp-text"><i class="fas fa-quote-right"></i> Top Recurring Phrases</div>
            <div class="pill-list">${ngramPills || '<span style="font-size:12px;color:var(--txt-dim);">No n-grams found.</span>'}</div>
        </div>
        <div class="result-card">
            <div class="result-title nlp-text"><i class="fas fa-tags"></i> Named Entities</div>
            <div class="pill-list">${entityHtml || '<span style="font-size:12px;color:var(--txt-dim);">No entities found.</span>'}</div>
        </div>
    `;
}

let aiVecMethod = 'tfidf';
let aiMaxFeatures = 300;

document.querySelectorAll('#ai-vec-method-picker .pick-chip').forEach(chip => {
    chip.onclick = () => {
        document.querySelectorAll('#ai-vec-method-picker .pick-chip').forEach(c => c.classList.remove('selected', 'nlp-sel'));
        chip.classList.add('selected', 'nlp-sel');
        aiVecMethod = chip.dataset.method;
    };
});

const aiMaxFeatSlider = document.getElementById('ai-maxfeat-slider');
if (aiMaxFeatSlider) {
    aiMaxFeatSlider.oninput = () => {
        aiMaxFeatures = parseInt(aiMaxFeatSlider.value, 10);
        document.getElementById('ai-maxfeat-value').textContent = aiMaxFeatures;
    };
}

document.getElementById('ai-vectorize-btn').onclick = async () => {
    const btn = document.getElementById('ai-vectorize-btn');
    const resultsArea = document.getElementById('ai-vectorize-results');

    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Vectorizing...';
    resultsArea.innerHTML = '<div class="ai-loading"><i class="fas fa-spinner fa-spin"></i> Fitting vectorizer...</div>';

    try {
        const formData = new FormData();
        formData.append('method', AI_STATE.mode === 'developer' ? aiVecMethod : 'tfidf');
        formData.append('max_features', String(AI_STATE.mode === 'developer' ? aiMaxFeatures : 300));

        const res = await fetch('/phase3/vectorize', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.status === 'success') {
            const termPills = (data.top_terms || []).map(t =>
                `<span class="ngram-pill">${t.term}<span class="count-badge">${t.score}</span></span>`
            ).join('');

            resultsArea.innerHTML = `
                <div class="result-card">
                    <div class="result-title nlp-text"><i class="fas fa-vector-square"></i> Vectorization Result (${data.method.toUpperCase()})</div>
                    <div class="tile-grid">
                        <div class="stat-tile nlp-tile"><div class="stat-value">${data.matrix_shape[0]}</div><div class="stat-label">Documents</div></div>
                        <div class="stat-tile nlp-tile"><div class="stat-value">${data.matrix_shape[1]}</div><div class="stat-label">Features</div></div>
                        <div class="stat-tile nlp-tile"><div class="stat-value">${data.vocabulary_size}</div><div class="stat-label">Vocabulary</div></div>
                    </div>
                    ${termPills ? `<p style="font-size:11px;color:var(--txt-dim);margin:18px 0 10px;text-transform:uppercase;letter-spacing:1px;">Top Weighted Terms</p><div class="pill-list">${termPills}</div>` : ''}
                </div>`;

            document.getElementById('ai-nlp-done-banner').style.display = 'block';
            showAiToast('Vectorization complete.');

            if (AI_STATE.mode === 'user') {
                autoContinue(document.getElementById('ai-nlp-done-banner'), 'Moving on to RAG — Ask Your Data', goToRagStage, 4);
            }
        } else {
            resultsArea.innerHTML = `<p style="color:var(--ai-red);">⚠️ ${data.message}</p>`;
            showAiToast(data.message, 'error');
        }
    } catch (e) {
        resultsArea.innerHTML = `<p style="color:var(--ai-red);">❌ Connection failed.</p>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-bolt"></i> Run Vectorization';
    }
};

document.getElementById('ai-continue-to-rag-btn').onclick = () => goToRagStage();

/* =====================================================================
   STAGE 3 — RAG
===================================================================== */

function goToRagStage() {
    setJourneyStage('nlp', 'done');
    markConnectorDone('nlp');
    goToMajorStage('rag');
    checkRagReadiness();
}

async function checkRagReadiness() {
    try {
        const res = await fetch('/phase3/rag/status');
        const data = await res.json();

        const existing = document.getElementById('ai-rag-config-warning');
        if (existing) existing.remove();

        if (data.status === 'success' && !data.llm_configured) {
            const warning = document.createElement('div');
            warning.id = 'ai-rag-config-warning';
            warning.style.cssText = 'max-width:600px;margin:0 auto 22px;padding:14px 20px;border-radius:12px;background:rgba(255,64,129,0.08);border:1px solid rgba(255,64,129,0.3);font-size:12px;color:var(--txt-dim);text-align:left;';
            warning.innerHTML = `
                <i class="fas fa-triangle-exclamation" style="color:var(--ai-red);"></i>
                <strong style="color:var(--ai-red);">Chatbot not configured yet.</strong>
                The server is missing its <code>ANTHROPIC_API_KEY</code>. You can still build the
                search index below, but answers won't generate until this is set
                (see the README's setup section) and the app is restarted.
            `;
            document.getElementById('ai-rag-status').insertAdjacentElement('beforebegin', warning);
        }

        // In User Mode, only auto-build the index if the chatbot can
        // actually answer afterward — otherwise auto-building it just
        // leads straight into a confusing dead end with no explanation.
        if (AI_STATE.mode === 'user' && (data.llm_configured || data.status !== 'success')) {
            document.getElementById('ai-build-index-btn').click();
        }
    } catch (e) {
        // If the status check itself fails, fall back to the previous
        // behavior rather than blocking the journey entirely.
        if (AI_STATE.mode === 'user') {
            document.getElementById('ai-build-index-btn').click();
        }
    }
}

const aiChunkSlider = document.getElementById('ai-chunksize-slider');
const aiOverlapSlider = document.getElementById('ai-overlap-slider');
const aiTopkSlider = document.getElementById('ai-topk-slider');

if (aiChunkSlider) aiChunkSlider.oninput = () => document.getElementById('ai-chunksize-value').textContent = aiChunkSlider.value;
if (aiOverlapSlider) aiOverlapSlider.oninput = () => document.getElementById('ai-overlap-value').textContent = aiOverlapSlider.value;
if (aiTopkSlider) aiTopkSlider.oninput = () => document.getElementById('ai-topk-value').textContent = aiTopkSlider.value;

document.getElementById('ai-build-index-btn').onclick = async () => {
    const btn = document.getElementById('ai-build-index-btn');
    const statusBox = document.getElementById('ai-rag-status');
    const statusText = document.getElementById('ai-rag-status-text');

    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Building Index...';
    statusText.textContent = 'Embedding and indexing chunks...';

    try {
        const formData = new FormData();
        formData.append('chunk_size', String(AI_STATE.mode === 'developer' && aiChunkSlider ? aiChunkSlider.value : 120));
        formData.append('chunk_overlap', String(AI_STATE.mode === 'developer' && aiOverlapSlider ? aiOverlapSlider.value : 20));
        formData.append('split_mode', 'token');

        const res = await fetch('/phase3/rag/build-index', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.status === 'success') {
            AI_STATE.ragReady = true;
            statusBox.classList.add('ready');
            statusText.textContent = `Index ready — ${data.chunk_count} chunks, ${data.embedding_dim}-dim embeddings`;
            document.getElementById('ai-chat-window').style.display = 'block';
            setJourneyStage('rag', 'current');
            showAiToast('RAG index built — your AI Pipeline is fully ready!');
            renderSuggestedQuestions();
        } else {
            statusText.textContent = 'Index build failed.';
            showAiToast(data.message, 'error');
        }
    } catch (e) {
        statusText.textContent = 'Connection failed.';
        showAiToast('Connection failed while building index.', 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-database"></i> Rebuild RAG Index';
    }
};

function renderSuggestedQuestions() {
    const messages = document.getElementById('ai-chat-messages');
    const existing = document.getElementById('ai-suggested-qs');
    if (existing) existing.remove();

    const suggestions = [
        'Summarize what this data is about',
        'What stands out the most?',
        'Are there any negative patterns?',
    ];
    const wrap = document.createElement('div');
    wrap.id = 'ai-suggested-qs';
    wrap.className = 'suggested-questions';
    wrap.innerHTML = suggestions.map(q => `<span class="suggested-q-chip">${q}</span>`).join('');
    messages.parentElement.insertBefore(wrap, messages);

    wrap.querySelectorAll('.suggested-q-chip').forEach(chip => {
        chip.onclick = () => {
            document.getElementById('ai-chat-input').value = chip.textContent;
            sendAiChatMessage();
        };
    });
}

function appendAiChatBubble(role, html) {
    const messages = document.getElementById('ai-chat-messages');
    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${role}`;
    bubble.innerHTML = html;
    messages.appendChild(bubble);
    messages.scrollTop = messages.scrollHeight;
    return bubble;
}

async function sendAiChatMessage() {
    const input = document.getElementById('ai-chat-input');
    const question = input.value.trim();
    if (!question || !AI_STATE.ragReady) return;

    const suggestions = document.getElementById('ai-suggested-qs');
    if (suggestions) suggestions.remove();

    appendAiChatBubble('user', question);
    input.value = '';

    const thinkingBubble = appendAiChatBubble('assistant', '<div class="typing-dots"><span></span><span></span><span></span></div>');

    try {
        const formData = new FormData();
        formData.append('question', question);
        formData.append('k', String(AI_STATE.mode === 'developer' && aiTopkSlider ? aiTopkSlider.value : 5));

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
            const isConfigError = (data.message || '').toLowerCase().includes('api_key') || (data.message || '').toLowerCase().includes('api key');
            thinkingBubble.innerHTML = isConfigError
                ? `⚠️ The chatbot isn't fully configured yet — the server needs an <code>ANTHROPIC_API_KEY</code> set. Ask whoever runs this app to add it and restart the server.`
                : `⚠️ ${data.message}`;
        }
    } catch (e) {
        thinkingBubble.innerHTML = '❌ Connection failed. Check your network and try again.';
    }
}

document.getElementById('ai-chat-send-btn').onclick = sendAiChatMessage;
document.getElementById('ai-chat-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendAiChatMessage();
});

/* ════════════════════════════════════════════
   INIT
════════════════════════════════════════════ */
window.onload = () => {
    applyAiModeUI();
    initAiUploadZone();
    tryLoadAiFromSession();
};
