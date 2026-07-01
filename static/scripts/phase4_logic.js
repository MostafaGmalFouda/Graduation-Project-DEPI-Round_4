/* =====================================================================
   BRight AI — Phase 4 Logic (Machine Learning)
   Handles: file upload, target selection, model recommend/train
            (User & Developer modes), results/plots, and predictions.
===================================================================== */

const P4_STATE = {
    mode: (typeof P4_MODE !== 'undefined') ? P4_MODE : 'user',
    target: null,
    taskType: null,
    columns: [],
    selectedModel: 'random_forest',
    trainResult: null,
    availableModels: { classification: [], regression: [], clustering: [] },
};

// ── Step navigation ──────────────────────────────────────────────────────
function unlockP4Step(step) {
    const btn = document.querySelector(`.step-btn.ml-step[data-step="${step}"]`);
    if (btn) btn.classList.remove('locked');
}

function goToP4Step(step) {
    document.querySelectorAll('.step-panel.ml-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.step-btn.ml-step').forEach(b => b.classList.remove('active'));
    document.getElementById('step-' + step).classList.add('active');
    const btn = document.querySelector(`.step-btn.ml-step[data-step="${step}"]`);
    if (btn) btn.classList.add('active');
}

document.querySelectorAll('.step-btn.ml-step').forEach(btn => {
    btn.addEventListener('click', () => {
        if (btn.classList.contains('locked')) return;
        goToP4Step(btn.dataset.step);
    });
});

function applyP4ModeUI() {
    const isDev = P4_STATE.mode === 'developer';
    document.getElementById('mode-title-ml').textContent =
        isDev ? 'Developer Mode — Full Control' : 'User Mode — Guided Flow';

    document.getElementById('p4-user-train-block').style.display = isDev ? 'none' : 'block';
    document.getElementById('p4-dev-train-block').style.display = isDev ? 'block' : 'none';
}

function showP4Toast(message, type = 'success') {
    const existing = document.querySelector('.p4-toast');
    if (existing) existing.remove();
    const toast = document.createElement('div');
    toast.className = `p4-toast ${type}`;
    const icon = type === 'error' ? 'fa-circle-exclamation' : 'fa-circle-check';
    toast.innerHTML = `<i class="fas ${icon}"></i> <span>${message}</span>`;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// ── STEP 1: Upload + target column ──────────────────────────────────────
function initP4UploadZone() {
    const zone = document.getElementById('p4-upload-zone');
    const input = document.getElementById('p4-file-input');

    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) {
            input.files = e.dataTransfer.files;
            handleP4FileSelected(e.dataTransfer.files[0]);
        }
    });
    input.addEventListener('change', () => {
        if (input.files.length) handleP4FileSelected(input.files[0]);
    });
}

async function handleP4FileSelected(file) {
    const label = document.getElementById('p4-file-name-label');
    label.textContent = file.name;
    label.style.display = 'inline-block';
    await detectP4Columns(file);
}

async function tryLoadP4FromSession() {
    await detectP4Columns(null);
}

async function detectP4Columns(file) {
    const formData = new FormData();
    if (file) formData.append('file', file);

    try {
        const res = await fetch('/phase4/detect-columns', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.status !== 'success') {
            if (file) showP4Toast(data.message || 'Could not read file.', 'error');
            return;
        }

        document.getElementById('p4-rows-cols').textContent = `${data.rows} rows × ${data.cols} cols`;
        P4_STATE.columns = data.columns;
        renderP4TargetPicker(data.columns);
        document.getElementById('p4-target-section').style.display = 'block';

        // Data is now loaded — whether just uploaded or already sitting in
        // the session from Phase 1 — so stop asking for a file. Hide the
        // upload zone and show a clear "loaded" confirmation instead.
        const zone = document.getElementById('p4-upload-zone');
        zone.style.display = 'none';

        let loadedBanner = document.getElementById('p4-loaded-banner');
        if (!loadedBanner) {
            loadedBanner = document.createElement('div');
            loadedBanner.id = 'p4-loaded-banner';
            loadedBanner.style.cssText = 'max-width:560px;margin:0 auto 22px;padding:12px 20px;border-radius:12px;background:rgba(255,64,129,0.06);border:1px solid var(--p4-panel-border);font-size:12px;color:var(--txt-dim);';
            zone.insertAdjacentElement('afterend', loadedBanner);
        }
        loadedBanner.style.display = 'block';
        loadedBanner.innerHTML = `
            <i class="fas fa-circle-check" style="color:var(--p4-green);"></i>
            <span>${file ? `Loaded "${file.name}"` : 'Using your data from Phase 1'} — ${data.rows} rows × ${data.cols} columns</span>
            <a id="p4-swap-file-link" style="margin-left:10px;color:var(--p4-accent);text-decoration:underline;cursor:pointer;">Use a different file</a>
        `;
        document.getElementById('p4-swap-file-link').onclick = () => {
            zone.style.display = 'block';
            loadedBanner.style.display = 'none';
            document.getElementById('p4-target-section').style.display = 'none';
        };

    } catch (e) {
        if (file) showP4Toast('Connection failed while detecting columns.', 'error');
    }
}

function renderP4TargetPicker(columns) {
    const wrap = document.getElementById('p4-target-picker');
    wrap.innerHTML = '';
    columns.forEach(col => {
        const chip = document.createElement('div');
        chip.className = 'target-chip';
        chip.innerHTML = `${col.name}<span class="chip-meta">${col.dtype} · ${col.n_unique}u</span>`;
        chip.onclick = () => {
            wrap.querySelectorAll('.target-chip').forEach(c => c.classList.remove('selected'));
            chip.classList.add('selected');
            P4_STATE.target = col.name;
            document.getElementById('p4-confirm-target-btn').disabled = false;
        };
        wrap.appendChild(chip);
    });
}

document.getElementById('p4-skip-target').onclick = () => {
    P4_STATE.target = null;
    document.querySelectorAll('.target-chip').forEach(c => c.classList.remove('selected'));
    document.getElementById('p4-confirm-target-btn').disabled = false;
    showP4Toast('Clustering mode — no target selected.');
};

document.getElementById('p4-confirm-target-btn').onclick = async () => {
    unlockP4Step('train');
    goToP4Step('train');

    if (P4_STATE.mode === 'developer') {
        await loadP4AvailableModels();
    } else {
        await loadP4Recommendations();
    }
};

// ── STEP 2a: USER MODE — recommendations ─────────────────────────────────
async function loadP4Recommendations() {
    const grid = document.getElementById('p4-recommend-grid');
    grid.innerHTML = '<div class="p4-loading"><i class="fas fa-spinner fa-spin"></i> Getting recommendations...</div>';

    try {
        const formData = new FormData();
        if (P4_STATE.target) formData.append('target', P4_STATE.target);

        const res = await fetch('/phase4/recommend', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.status === 'success') {
            P4_STATE.taskType = data.task_type;
            grid.innerHTML = data.recommendations.map((r, i) => `
                <div class="recommend-card ${i === 0 ? 'top-pick' : ''}">
                    ${i === 0 ? '<div class="rank-badge">TOP PICK</div>' : ''}
                    <div class="model-name"><i class="fas fa-cube"></i> ${r.model_name.replace(/_/g, ' ')}</div>
                    <div class="model-reason">${r.reason}</div>
                </div>
            `).join('');
        } else {
            grid.innerHTML = `<p style="color:var(--p4-accent);">⚠️ ${data.message}</p>`;
        }
    } catch (e) {
        grid.innerHTML = `<p style="color:var(--p4-accent);">❌ Connection failed.</p>`;
    }
}

document.getElementById('p4-user-train-btn').onclick = async () => {
    await runP4Train('user');
};

// ── STEP 2b: DEVELOPER MODE — manual model + params ──────────────────────
// CRITICAL FIX: resolve the REAL task_type (via /phase4/recommend) before
// rendering the model grid, and only show models compatible with it.
// Previously this merged classification + regression + clustering model
// names into one list, so picking e.g. "knn" on a continuous (regression)
// target silently built a KNeighborsClassifier and crashed training with
// "Unknown label type: continuous".
async function loadP4AvailableModels() {
    try {
        const [modelsRes, taskRes] = await Promise.all([
            fetch('/phase4/available-models'),
            fetch('/phase4/recommend', {
                method: 'POST',
                body: (() => { const fd = new FormData(); if (P4_STATE.target) fd.append('target', P4_STATE.target); return fd; })(),
            }),
        ]);

        const modelsData = await modelsRes.json();
        const taskData = await taskRes.json();

        if (modelsData.status === 'success') P4_STATE.availableModels = modelsData.models;
        P4_STATE.taskType = (taskData.status === 'success') ? taskData.task_type : 'classification';

        const compatible = P4_STATE.availableModels[P4_STATE.taskType] || [];
        if (!compatible.includes(P4_STATE.selectedModel)) {
            P4_STATE.selectedModel = compatible[0] || P4_STATE.selectedModel;
        }

        renderP4ModelGrid();
    } catch (e) {
        showP4Toast('Could not load model list.', 'error');
    }
}

function renderP4ModelGrid() {
    const grid = document.getElementById('p4-model-select-grid');
    const compatible = P4_STATE.availableModels[P4_STATE.taskType] || [];

    let taskBanner = document.getElementById('p4-task-type-banner');
    if (!taskBanner) {
        taskBanner = document.createElement('p');
        taskBanner.id = 'p4-task-type-banner';
        taskBanner.style.cssText = 'font-size:11px;color:var(--txt-dim);margin-bottom:12px;';
        grid.parentElement.insertBefore(taskBanner, grid);
    }
    taskBanner.innerHTML = `<i class="fas fa-bullseye" style="color:var(--p4-accent);"></i> Detected task: <strong style="color:var(--p4-accent);">${P4_STATE.taskType}</strong> — only compatible models are shown.`;

    grid.innerHTML = '';
    compatible.forEach(name => {
        const opt = document.createElement('div');
        opt.className = 'model-select-option' + (name === P4_STATE.selectedModel ? ' selected' : '');
        opt.textContent = name.replace(/_/g, ' ');
        opt.onclick = () => {
            grid.querySelectorAll('.model-select-option').forEach(o => o.classList.remove('selected'));
            opt.classList.add('selected');
            P4_STATE.selectedModel = name;
            renderP4ParamFields(name);
        };
        grid.appendChild(opt);
    });
    renderP4ParamFields(P4_STATE.selectedModel);
}

const MODEL_PARAM_HINTS = {
    random_forest: [{ key: 'n_estimators', label: 'n_estimators', placeholder: '100' }, { key: 'max_depth', label: 'max_depth (blank = none)', placeholder: '' }],
    logistic_regression: [{ key: 'C', label: 'C (inverse regularization)', placeholder: '1.0' }, { key: 'max_iter', label: 'max_iter', placeholder: '1000' }],
    linear_regression: [],
    svm: [{ key: 'C', label: 'C', placeholder: '1.0' }, { key: 'kernel', label: 'kernel (rbf/linear/poly)', placeholder: 'rbf' }],
    knn: [{ key: 'n_neighbors', label: 'n_neighbors', placeholder: '5' }],
    decision_tree: [{ key: 'max_depth', label: 'max_depth (blank = none)', placeholder: '' }],
    kmeans: [{ key: 'n_clusters', label: 'n_clusters', placeholder: '3' }],
    dbscan: [{ key: 'eps', label: 'eps', placeholder: '0.5' }, { key: 'min_samples', label: 'min_samples', placeholder: '5' }],
};

function renderP4ParamFields(modelName) {
    const container = document.getElementById('p4-params-fields');
    const fields = MODEL_PARAM_HINTS[modelName] || [];
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

document.getElementById('p4-tuning-switch').onclick = function () {
    const isOn = this.classList.toggle('on');
    document.getElementById('p4-tuning-options').style.display = isOn ? 'block' : 'none';
};

document.querySelectorAll('[data-tune]').forEach(opt => {
    opt.onclick = () => {
        document.querySelectorAll('[data-tune]').forEach(o => o.classList.remove('selected'));
        opt.classList.add('selected');
    };
});

const p4CvSlider = document.getElementById('p4-cv-slider');
if (p4CvSlider) p4CvSlider.oninput = () => document.getElementById('p4-cv-value').textContent = p4CvSlider.value;

const p4TestSizeSlider = document.getElementById('p4-testsize-slider');
if (p4TestSizeSlider) {
    p4TestSizeSlider.oninput = () => {
        document.getElementById('p4-testsize-value').textContent = Math.round(parseFloat(p4TestSizeSlider.value) * 100) + '%';
    };
}

document.getElementById('p4-dev-train-btn').onclick = async () => {
    await runP4Train('developer');
};

// ── Shared training call ─────────────────────────────────────────────────
async function runP4Train(mode) {
    const statusBox = document.getElementById('p4-training-status');
    statusBox.style.display = 'flex';

    const trainBtn = mode === 'user'
        ? document.getElementById('p4-user-train-btn')
        : document.getElementById('p4-dev-train-btn');
    trainBtn.disabled = true;

    try {
        const formData = new FormData();
        formData.append('run_mode', mode);
        if (P4_STATE.target) formData.append('target', P4_STATE.target);

        if (mode === 'developer') {
            formData.append('model_name', P4_STATE.selectedModel);

            const params = {};
            document.querySelectorAll('#p4-params-fields input[data-param]').forEach(input => {
                if (input.value.trim() !== '') {
                    const num = Number(input.value);
                    params[input.dataset.param] = isNaN(num) ? input.value.trim() : num;
                }
            });
            formData.append('params', JSON.stringify(params));

            const tuningOn = document.getElementById('p4-tuning-switch').classList.contains('on');
            if (tuningOn) {
                const method = document.querySelector('[data-tune].selected')?.dataset.tune || 'grid';
                // Minimal, safe default search space per model family
                const paramSpace = buildDefaultParamSpace(P4_STATE.selectedModel);
                formData.append('tuning', JSON.stringify({ method, param_space: paramSpace }));
            }

            formData.append('cv', p4CvSlider ? p4CvSlider.value : '5');
            formData.append('test_size', p4TestSizeSlider ? p4TestSizeSlider.value : '0.2');
        }

        const res = await fetch('/phase4/train', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.status === 'success') {
            P4_STATE.trainResult = data.result;
            P4_STATE.taskType = data.result.task_type;
            renderP4Results(data.result);
            unlockP4Step('results');
            unlockP4Step('predict');
            renderP4PredictForm();
            goToP4Step('results');
            showP4Toast('Model trained successfully!');
        } else {
            showP4Toast(data.message || 'Training failed.', 'error');
        }
    } catch (e) {
        showP4Toast('Connection failed during training.', 'error');
    } finally {
        statusBox.style.display = 'none';
        trainBtn.disabled = false;
    }
}

function buildDefaultParamSpace(modelName) {
    const spaces = {
        random_forest: { n_estimators: [50, 100, 200], max_depth: [5, 10, null] },
        logistic_regression: { C: [0.1, 1.0, 10.0] },
        svm: { C: [0.1, 1.0, 10.0] },
        knn: { n_neighbors: [3, 5, 7, 9] },
        decision_tree: { max_depth: [3, 5, 10, null] },
    };
    return spaces[modelName] || {};
}

// ── STEP 3: Results & plots ───────────────────────────────────────────────
function renderP4Results(result) {
    const summaryBar = document.getElementById('p4-model-summary');
    const metricsGrid = document.getElementById('p4-metrics-grid');
    const tuningBox = document.getElementById('p4-tuning-summary');

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
            ${metricTile(m.accuracy, 'Accuracy')}
            ${metricTile(m.precision, 'Precision')}
            ${metricTile(m.recall, 'Recall')}
            ${metricTile(m.f1_score, 'F1 Score')}
            ${m.roc_auc != null ? metricTile(m.roc_auc, 'ROC-AUC') : ''}
        `;
    } else if (result.task_type === 'regression') {
        tiles = `
            ${metricTile(m.r2, 'R²')}
            ${metricTile(m.mae, 'MAE', false)}
            ${metricTile(m.rmse, 'RMSE', false)}
            ${metricTile(m.mse, 'MSE', false)}
        `;
    } else {
        tiles = `<p style="font-size:12px;color:var(--txt-dim);grid-column:1/-1;">${(m.note || 'Clustering complete.')}</p>`;
    }
    metricsGrid.innerHTML = tiles;

    if (result.tuning_results) {
        const tr = result.tuning_results;
        tuningBox.style.display = 'block';
        tuningBox.innerHTML = `
            <div class="result-title" style="color:var(--p4-accent);font-size:12px;margin-bottom:14px;text-transform:uppercase;letter-spacing:1px;">
                <i class="fas fa-magnifying-glass-chart"></i> Hyperparameter Tuning Result
            </div>
            <p style="font-size:12px;color:var(--txt-dim);">Best Score: <strong style="color:var(--p4-accent);">${tr.best_score?.toFixed ? tr.best_score.toFixed(4) : tr.best_score}</strong></p>
            <p style="font-size:12px;color:var(--txt-dim);margin-top:6px;">Best Params: <code>${JSON.stringify(tr.best_params)}</code></p>
        `;
    } else {
        tuningBox.style.display = 'none';
    }

    renderP4PlotButtons(result.task_type);
}

function metricTile(value, label, isPct = true) {
    const display = (value == null) ? '—' : (isPct ? (value * 100).toFixed(1) + '%' : value.toFixed ? value.toFixed(3) : value);
    return `<div class="metric-tile"><div class="metric-value">${display}</div><div class="metric-label">${label}</div></div>`;
}

function renderP4PlotButtons(taskType) {
    const buttonsRow = document.getElementById('p4-plot-buttons');
    const plots = taskType === 'classification'
        ? [['confusion_matrix', 'Confusion Matrix'], ['roc_curve', 'ROC Curve'], ['feature_importance', 'Feature Importance']]
        : taskType === 'regression'
        ? [['actual_vs_predicted', 'Actual vs Predicted'], ['feature_importance', 'Feature Importance'], ['correlation_matrix', 'Correlation Matrix']]
        : [];

    buttonsRow.innerHTML = plots.map(([type, label]) =>
        `<button class="plot-pick-btn" data-plot="${type}">${label}</button>`
    ).join('');

    buttonsRow.querySelectorAll('.plot-pick-btn').forEach(btn => {
        btn.onclick = () => generateP4Plot(btn.dataset.plot, btn);
    });
}

async function generateP4Plot(plotType, btnEl) {
    const gallery = document.getElementById('p4-plot-gallery');
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
                <img src="${data.view_url}" onclick="openP4Lightbox('${data.view_url}')">
            `;
            gallery.prepend(card);
        } else {
            showP4Toast(data.message || 'Could not generate plot.', 'error');
        }
    } catch (e) {
        showP4Toast('Connection failed while generating plot.', 'error');
    } finally {
        if (btnEl) { btnEl.disabled = false; btnEl.innerHTML = originalLabel; }
    }
}

function openP4Lightbox(url) {
    document.getElementById('p4-lightbox-img').src = url;
    document.getElementById('p4-lightbox').style.display = 'block';
}
function closeP4Lightbox() {
    document.getElementById('p4-lightbox').style.display = 'none';
}

document.getElementById('p4-goto-predict-btn').onclick = () => goToP4Step('predict');

// ── STEP 4: Predict ────────────────────────────────────────────────────────
function renderP4PredictForm() {
    const form = document.getElementById('p4-predict-form');
    const featureCols = P4_STATE.columns
        .map(c => c.name)
        .filter(name => name !== P4_STATE.target);

    form.innerHTML = featureCols.map(name => `
        <div class="predict-field">
            <label>${name}</label>
            <input type="text" data-feature="${name}" placeholder="value">
        </div>
    `).join('');
}

document.getElementById('p4-predict-btn').onclick = async () => {
    const resultBox = document.getElementById('p4-prediction-result');
    const sample = {};
    document.querySelectorAll('#p4-predict-form input[data-feature]').forEach(input => {
        const num = Number(input.value);
        sample[input.dataset.feature] = (input.value.trim() !== '' && !isNaN(num)) ? num : input.value;
    });

    resultBox.innerHTML = '<div class="p4-loading"><i class="fas fa-spinner fa-spin"></i> Predicting...</div>';

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
                    ${probsHtml ? `<div class="contrib-list"><p style="font-size:10px;color:var(--txt-dim);text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">Class Probabilities</p>${probsHtml}</div>` : ''}
                    ${contribHtml ? `<div class="contrib-list"><p style="font-size:10px;color:var(--txt-dim);text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">Top Contributing Features</p>${contribHtml}</div>` : ''}
                </div>
            `;
        } else {
            resultBox.innerHTML = `<p style="color:var(--p4-accent);">⚠️ ${data.message}</p>`;
        }
    } catch (e) {
        resultBox.innerHTML = `<p style="color:var(--p4-accent);">❌ Connection failed.</p>`;
    }
};

// ── Init ────────────────────────────────────────────────────────────────────
window.onload = () => {
    applyP4ModeUI();
    initP4UploadZone();
    tryLoadP4FromSession();
};
