/* ============================================================
   BRight AI — Phase 5: ML Logic
   ============================================================ */

const ML = { columns: [], guessedTarget: null, availableModels: {}, hyperparamSpecs: {}, lastResult: null, recommendation: null, recommendedModel: null, predictSchema: null, allFeatures: null };

// Fetches the full logical feature schema (original names/categories/ranges
// for every column in the dataset) ONCE and caches it, instead of asking the
// server for a specific subset of columns via a long '?columns=a,b,c,...'
// query string. With a wide one-hot encoded dataset that list can get long
// enough to be silently truncated, which used to come back as "only some of
// my original features show up". Fetching everything once and filtering
// client-side removes that failure mode entirely.
async function ensureFeatureSchema() {
    if (ML.allFeatures) return ML.allFeatures;
    try {
        const res = await fetch("/ml/feature_schema");
        const data = await res.json();
        if (data.status === "success") {
            ML.allFeatures = data.features;
            return ML.allFeatures;
        }
    } catch (e) {
        // fall through
    }
    return null;
}

function renderHyperparamControls(taskType, modelName) {
    const container = document.getElementById("ml-dev-hyperparams");
    if (!container) return;

    const specs = (ML.hyperparamSpecs[taskType] && ML.hyperparamSpecs[taskType][modelName]) || [];

    if (specs.length === 0) {
        container.innerHTML = `<p class="hint" style="margin:6px 0 10px;">This model has no extra hyperparameters to tune.</p>`;
    } else {
        container.innerHTML = specs.map(spec => {
            const inputId = `ml-hp-${spec.name}`;
            if (spec.type === "select") {
                const opts = spec.options.map(o =>
                    `<option value="${o}" ${o === spec.default ? "selected" : ""}>${o}</option>`
                ).join("");
                return `
                    <div class="form-group">
                        <label>${spec.label}</label>
                        <select id="${inputId}" class="form-select ml-hyperparam-input" data-name="${spec.name}" data-type="select">${opts}</select>
                    </div>
                `;
            }
            if (spec.type === "bool") {
                return `
                    <div class="form-check">
                        <input type="checkbox" id="${inputId}" class="ml-hyperparam-input" data-name="${spec.name}" data-type="bool" ${spec.default ? "checked" : ""}>
                        <label for="${inputId}">${spec.label}</label>
                    </div>
                `;
            }
            const step = spec.type === "float" ? (spec.step || 0.01) : (spec.step || 1);
            const placeholder = (spec.default === null || spec.default === undefined) ? "auto" : spec.default;
            return `
                <div class="form-group">
                    <label>${spec.label}</label>
                    <input type="number" id="${inputId}" class="form-select ml-hyperparam-input"
                        data-name="${spec.name}" data-type="${spec.type}"
                        min="${spec.min ?? ""}" max="${spec.max ?? ""}" step="${step}" placeholder="e.g. ${placeholder}">
                </div>
            `;
        }).join("");
    }

    // Gentle scaling nudge — informational only, never forces the choice.
    const scaleHint = document.getElementById("ml-dev-scale-hint");
    const scaleBox = document.getElementById("ml-dev-scale");
    const benefitsFromScaling = ["svm", "knn", "logistic_regression"].includes(modelName);
    if (scaleHint) {
        scaleHint.textContent = benefitsFromScaling
            ? "(recommended for this model)"
            : "(usually not needed for tree-based models, but won't hurt)";
    }
    if (scaleBox) scaleBox.checked = benefitsFromScaling;
}

async function fetchTargetOptions(allColumns) {
    const features = await ensureFeatureSchema();
    if (!features) return null;
    // A One-Hot group (column_by_category) can't serve as a single
    // scalar target — only plain numeric or Label-Encoded columns can.
    return features.filter(f => !f.column_by_category);
}

function populateTargetSelect(selectEl, options, allColumns) {
    const list = options || allColumns.map(c => ({ name: c, clean_columns: [c] }));
    selectEl.innerHTML =
        `<option value="" selected disabled>-- Choose a target column --</option>` +
        list.map(f => `<option value="${f.clean_columns[0]}">${f.name}</option>`).join("");
}

async function initML() {
    document.getElementById("mode-title").textContent =
        APP_MODE === "developer"
            ? "Developer Mode — pick your model, features & hyperparameters"
            : "User Mode — one-click AutoML";

    try {
        const res = await fetch("/ml/status");
        const data = await res.json();

        if (data.status !== "ready") {
            document.getElementById("ml-no-data").style.display = "block";
            return;
        }

        ML.columns = data.columns;
        ML.guessedTarget = data.guessed_target;
        ML.availableModels = data.available_models;
        ML.hyperparamSpecs = data.hyperparam_specs || {};

        // Transparency: tell the person which of THEIR original uploaded
        // columns are simply not present anymore (dropped upstream in
        // Phase 1 — excluded, treated as an ID column, or dropped as
        // high-cardinality free text) so a missing feature is never a
        // silent mystery.
        const missingNotice = document.getElementById("ml-missing-columns-notice");
        if (missingNotice) {
            if (data.missing_raw_columns && data.missing_raw_columns.length > 0) {
                missingNotice.style.display = "block";
                missingNotice.innerHTML = `<i class="fas fa-circle-info"></i> ${data.missing_raw_columns.length} of your original column(s) ` +
                    `aren't available here because Phase 1 preprocessing removed them ` +
                    `(excluded, treated as an ID column, or dropped as free text): ` +
                    `<b>${data.missing_raw_columns.join(", ")}</b>. Every other original column is still here — ` +
                    `just shown by its real name/categories, not the encoded values.`;
            } else {
                missingNotice.style.display = "none";
            }
        }

        document.getElementById("ml-workspace").style.display = "block";

        if (APP_MODE === "developer") {
            document.getElementById("ml-developer-mode").style.display = "block";
            await initDeveloperForm(data);
        } else {
            document.getElementById("ml-user-mode").style.display = "block";
            document.getElementById("ml-meta-rows").textContent = `${data.rows} rows`;
            document.getElementById("ml-meta-cols").textContent = `${data.cols} columns`;

            const targetSelect = document.getElementById("ml-user-target");
            const targetOptions = await fetchTargetOptions(data.columns);
            populateTargetSelect(targetSelect, targetOptions, data.columns);

            const badge = document.getElementById("ml-guessed-badge");
            if (data.guessed_target) {
                badge.textContent = `Suggested: ${data.guessed_target} (click to use)`;
                badge.style.display = "inline-block";
                badge.style.cursor = "pointer";
                badge.onclick = () => { targetSelect.value = data.guessed_target; };
            } else {
                badge.style.display = "none";
            }
        }
    } catch (e) {
        showToast("Could not reach the server: " + e.message, "error");
    }
}

async function initDeveloperForm(data) {
    const targetSelect = document.getElementById("ml-dev-target");
    const targetOptions = await fetchTargetOptions(data.columns);
    populateTargetSelect(targetSelect, targetOptions, data.columns);

    const devBadge = document.getElementById("ml-dev-guessed-badge");
    if (devBadge) {
        if (data.guessed_target) {
            devBadge.textContent = `Suggested: ${data.guessed_target} (click to use)`;
            devBadge.style.display = "inline-block";
            devBadge.style.cursor = "pointer";
            devBadge.onclick = () => { targetSelect.value = data.guessed_target; targetSelect.dispatchEvent(new Event("change")); };
        } else {
            devBadge.style.display = "none";
        }
    }

    const taskSelect = document.getElementById("ml-dev-task");
    const modelSelect = document.getElementById("ml-dev-model");

    function currentTask() {
        return taskSelect.value || data.guessed_task_type;
    }

    function refreshModelOptions() {
        const task = currentTask();
        const models = ML.availableModels[task] || ML.availableModels.classification;
        modelSelect.innerHTML = models.map(m =>
            `<option value="${m}">${m.replace(/_/g, " ")}</option>`
        ).join("");
        renderHyperparamControls(task, modelSelect.value);
    }
    taskSelect.addEventListener("change", refreshModelOptions);
    modelSelect.addEventListener("change", () => renderHyperparamControls(currentTask(), modelSelect.value));
    refreshModelOptions();

    async function refreshFeatureChecklist() {
        const target = targetSelect.value;
        const container = document.getElementById("ml-dev-features");

        if (!target) {
            container.innerHTML = `<p style="color:var(--txt-dim); font-size:12.5px;">Choose a target column above to see available features.</p>`;
            return;
        }

        const candidateColumns = data.columns.filter(c => c !== target);
        container.innerHTML = `<p style="color:var(--txt-dim); font-size:12.5px;">Loading features…</p>`;

        const allFeatures = await ensureFeatureSchema();

        if (!allFeatures) {
            container.innerHTML = candidateColumns.map(c => `
                <label class="col-chip categorical" style="cursor:pointer;">
                    <input type="checkbox" class="ml-feature-check" data-columns="${c}" checked style="margin-right:6px;">${c}
                </label>
            `).join("");
            return;
        }

        // Every logical feature EXCEPT the one whose encoded column(s)
        // include the chosen target — this is the full original feature
        // set, not a server-side subset, so nothing is ever missing here
        // just because a request happened to leave a column out.
        const features = allFeatures.filter(f => !f.clean_columns.includes(target));

        // One entry per ORIGINAL column — a One-Hot group (e.g.
        // Gender_Male/Gender_Female) shows up once as "Gender",
        // and checking/unchecking it toggles all its encoded columns.
        container.innerHTML = features.map(f => `
            <label class="col-chip ${f.type}" style="cursor:pointer;">
                <input type="checkbox" class="ml-feature-check" data-columns="${f.clean_columns.join(",")}" checked style="margin-right:6px;">${f.name}
                ${f.type === "categorical" ? `<span class="hint">(${f.options.length} categories)</span>` : ""}
            </label>
        `).join("");
    }
    targetSelect.addEventListener("change", refreshFeatureChecklist);
    refreshFeatureChecklist();
}

document.getElementById("ml-auto-recommend-btn")?.addEventListener("click", getRecommendation);
document.getElementById("ml-accept-recommend-btn")?.addEventListener("click", () => trainML(true, ML.recommendedModel));
document.getElementById("ml-choose-another-btn")?.addEventListener("click", toggleAltPicker);
document.getElementById("ml-train-alt-btn")?.addEventListener("click", () => {
    const alt = document.getElementById("ml-alt-model-select").value;
    if (alt) trainML(true, alt);
});
document.getElementById("ml-dev-run-btn")?.addEventListener("click", () => trainML(false));

async function getRecommendation() {
    const btn = document.getElementById("ml-auto-recommend-btn");
    const target = document.getElementById("ml-user-target").value;

    if (!target) {
        showToast("Please choose a target column first.", "error");
        return;
    }

    btn.disabled = true;
    document.getElementById("ml-recommend-card").style.display = "none";
    document.getElementById("ml-alt-model-picker").style.display = "none";
    document.getElementById("ml-results").style.display = "none";

    const formData = new FormData();
    formData.append("target_column", target);

    try {
        const res = await fetch("/ml/recommend", { method: "POST", body: formData });
        const data = await res.json();
        btn.disabled = false;

        if (data.status !== "success") {
            showToast(data.message || "Could not compute a recommendation.", "error");
            return;
        }

        ML.recommendation = data;
        ML.recommendedModel = data.recommended_model;
        renderRecommendation(data);
    } catch (e) {
        btn.disabled = false;
        showToast("Request failed: " + e.message, "error");
    }
}

function renderRecommendation(data) {
    const best = data.candidates.find(c => c.model === data.recommended_model) || data.candidates[0];
    const scoreLabel = data.task_type === "classification"
        ? `CV accuracy: ${(best.score * 100).toFixed(1)}%`
        : `CV R² score: ${best.score.toFixed(3)}`;

    document.getElementById("ml-recommend-name").textContent = best.model.replace(/_/g, " ");
    document.getElementById("ml-recommend-score").textContent = scoreLabel;

    const list = document.getElementById("ml-recommend-candidates");
    list.innerHTML = data.candidates.map(c => `
        <div class="candidate-row ${c.model === data.recommended_model ? "is-best" : ""}">
            <span class="candidate-name">${c.model.replace(/_/g, " ")}</span>
            <span class="candidate-score">${c.score <= -1e8 ? "n/a" : c.score.toFixed(3)}</span>
        </div>
    `).join("");

    const altSelect = document.getElementById("ml-alt-model-select");
    altSelect.innerHTML = data.candidates
        .filter(c => c.model !== data.recommended_model)
        .map(c => `<option value="${c.model}">${c.model.replace(/_/g, " ")}</option>`)
        .join("");

    document.getElementById("ml-alt-model-picker").style.display = "none";
    document.getElementById("ml-recommend-card").style.display = "block";
}

function toggleAltPicker() {
    const picker = document.getElementById("ml-alt-model-picker");
    picker.style.display = picker.style.display === "none" ? "block" : "none";
}

async function trainML(auto, modelName) {
    const loading = document.getElementById("ml-loading");
    const results = document.getElementById("ml-results");

    const targetFieldId = auto ? "ml-user-target" : "ml-dev-target";
    const target = document.getElementById(targetFieldId).value;
    if (!target) {
        showToast("Please choose a target column first.", "error");
        return;
    }

    loading.style.display = "flex";
    loading.classList.add("active");
    results.style.display = "none";

    const formData = new FormData();
    formData.append("mode", auto ? "user" : "developer");
    formData.append("target_column", target);

    if (auto) {
        if (modelName) formData.append("model_name", modelName);
    } else {
        formData.append("model_name", document.getElementById("ml-dev-model").value);
        const task = document.getElementById("ml-dev-task").value;
        if (task) formData.append("task_type", task);
        formData.append("test_size", document.getElementById("ml-dev-testsize").value);
        formData.append("scale", document.getElementById("ml-dev-scale").checked ? "true" : "false");

        document.querySelectorAll(".ml-hyperparam-input").forEach(inp => {
            const name = inp.dataset.name;
            if (inp.dataset.type === "bool") {
                formData.append(name, inp.checked ? "true" : "false");
            } else if (inp.value !== "") {
                formData.append(name, inp.value);
            }
        });

        document.querySelectorAll(".ml-feature-check:checked").forEach(cb => {
            cb.dataset.columns.split(",").forEach(col => formData.append("feature_columns", col));
        });
    }

    try {
        const res = await fetch("/ml/train", { method: "POST", body: formData });
        const data = await res.json();
        loading.style.display = "none";
        loading.classList.remove("active");

        if (data.status !== "success") {
            showToast(data.message || "Training failed.", "error");
            return;
        }

        ML.lastResult = data;
        renderMLResults(data);
        results.style.display = "block";
        showToast(`Trained ${data.model_name} — ready.`, "success");
    } catch (e) {
        loading.style.display = "none";
        loading.classList.remove("active");
        showToast("Request failed: " + e.message, "error");
    }
}

function renderMLResults(data) {
    const metrics = data.metrics;
    const cards = document.getElementById("ml-metric-cards");

    // User Mode: tell them whether they trained the recommended model or a
    // manually-chosen alternative from the "Choose Another Model" list.
    const noteBox = document.getElementById("ml-mode-note");
    if (noteBox) {
        if (data.mode === "user" && ML.recommendedModel) {
            const isRecommended = data.model_name === ML.recommendedModel;
            noteBox.style.display = "block";
            noteBox.innerHTML = isRecommended
                ? `<i class="fas fa-star"></i> Trained the <b>recommended</b> model: ${data.model_name.replace(/_/g, " ")}`
                : `<i class="fas fa-list"></i> Trained <b>${data.model_name.replace(/_/g, " ")}</b> (recommended was ${ML.recommendedModel.replace(/_/g, " ")})`;
        } else {
            noteBox.style.display = "none";
        }
    }

    if (metrics.task_type === "classification") {
        cards.innerHTML = `
            <div class="metric-card"><div class="metric-value">${(metrics.accuracy * 100).toFixed(1)}%</div><div class="metric-label">Accuracy</div></div>
            <div class="metric-card"><div class="metric-value">${metrics.f1_score}</div><div class="metric-label">F1 Score</div></div>
            <div class="metric-card"><div class="metric-value">${metrics.precision}</div><div class="metric-label">Precision</div></div>
            <div class="metric-card"><div class="metric-value">${metrics.recall}</div><div class="metric-label">Recall</div></div>
        `;
    } else {
        cards.innerHTML = `
            <div class="metric-card"><div class="metric-value">${metrics.r2_score}</div><div class="metric-label">R² Score</div></div>
            <div class="metric-card"><div class="metric-value">${metrics.rmse}</div><div class="metric-label">RMSE</div></div>
            <div class="metric-card"><div class="metric-value">${metrics.mae}</div><div class="metric-label">MAE</div></div>
            <div class="metric-card"><div class="metric-value">${metrics.mse}</div><div class="metric-label">MSE</div></div>
        `;
    }

    const plotsGrid = document.getElementById("ml-plots-grid");
    plotsGrid.innerHTML = "";
    for (const [key, filename] of Object.entries(data.plots)) {
        plotsGrid.innerHTML += plotCard(key, `/ml/view/${filename}`, `/ml/download/${filename}`);
    }

    // Build the "try a prediction" form using the ORIGINAL feature identity
    // (categorical dropdowns with real category names, numeric inputs with
    // a real observed range) instead of raw encoded columns.
    renderPredictForm(data.feature_columns);
}

function round2(n) {
    return Math.round(n * 100) / 100;
}

async function renderPredictForm(featureColumns) {
    const predictForm = document.getElementById("ml-predict-form");
    predictForm.innerHTML = `<p style="color:var(--txt-dim); font-size:13px;">Loading feature info…</p>`;

    const allFeatures = await ensureFeatureSchema();
    // Scope the cached full schema down to whatever this particular
    // trained model actually uses (its exact encoded feature_columns).
    const features = allFeatures
        ? allFeatures.filter(f => f.clean_columns.some(c => featureColumns.includes(c)))
        : null;

    if (!features) {
        // Fallback: plain numeric inputs, same as before.
        ML.predictSchema = null;
        predictForm.innerHTML = featureColumns.map(f => `
            <div class="chart-config-card" style="padding:12px 16px;">
                <div class="form-group" style="margin-bottom:0;">
                    <label>${f}</label>
                    <input type="number" step="any" class="form-select ml-predict-input" data-name="${f}" placeholder="value">
                </div>
            </div>
        `).join("");
        return;
    }

    ML.predictSchema = features;
    predictForm.innerHTML = features.map(f => {
        if (f.type === "categorical") {
            const opts = f.options.map(opt => `<option value="${opt}">${opt}</option>`).join("");
            return `
                <div class="chart-config-card" style="padding:12px 16px;">
                    <div class="form-group" style="margin-bottom:0;">
                        <label>${f.name}</label>
                        <select class="form-select ml-predict-input" data-name="${f.name}">${opts}</select>
                    </div>
                </div>
            `;
        }
        const hasRange = f.min !== null && f.min !== undefined && f.max !== null && f.max !== undefined;
        const hint = hasRange ? `<span class="hint">range: ${round2(f.min)} – ${round2(f.max)}</span>` : "";
        return `
            <div class="chart-config-card" style="padding:12px 16px;">
                <div class="form-group" style="margin-bottom:0;">
                    <label>${f.name} ${hint}</label>
                    <input type="number" step="any" class="form-select ml-predict-input" data-name="${f.name}" placeholder="value">
                </div>
            </div>
        `;
    }).join("");
}

document.getElementById("ml-predict-btn")?.addEventListener("click", async () => {
    if (!ML.lastResult) return;
    const row = {};
    const schema = ML.predictSchema;

    document.querySelectorAll(".ml-predict-input").forEach(inp => {
        const name = inp.dataset.name;
        const feat = schema ? schema.find(f => f.name === name) : null;

        if (!feat) {
            // No schema available (fallback mode) — send as-is.
            row[name] = inp.value || 0;
            return;
        }

        if (feat.type === "categorical" && feat.value_map) {
            // Label-Encoded: translate the chosen original label back to
            // the numeric code the model was actually trained on.
            const selectedLabel = inp.value;
            const code = Object.keys(feat.value_map).find(k => feat.value_map[k] === selectedLabel);
            row[feat.clean_columns[0]] = code !== undefined ? code : 0;
        } else if (feat.type === "categorical" && feat.column_by_category) {
            // One-Hot group: set the chosen category's column to 1, rest to 0.
            feat.clean_columns.forEach(col => { row[col] = 0; });
            const targetCol = feat.column_by_category[inp.value];
            if (targetCol) row[targetCol] = 1;
        } else {
            row[feat.clean_columns[0]] = inp.value || 0;
        }
    });

    try {
        const res = await fetch("/ml/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model_id: ML.lastResult.model_id, row }),
        });
        const data = await res.json();
        const box = document.getElementById("ml-predict-result");

        if (data.status !== "success") {
            box.className = "predict-result show";
            box.innerHTML = `⚠️ ${data.message || "Prediction failed."}`;
            return;
        }

        let extra = "";
        if (data.probabilities) {
            extra = "<br>" + Object.entries(data.probabilities)
                .map(([k, v]) => `${k}: ${(v * 100).toFixed(1)}%`).join(" · ");
        }
        box.className = "predict-result show";
        box.innerHTML = `Predicted <b>${data.target_column}</b>: <span class="predict-value">${data.prediction}</span>${extra}`;
    } catch (e) {
        showToast("Prediction request failed: " + e.message, "error");
    }
});

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
    let toast = document.getElementById("ml-toast");
    if (!toast) {
        toast = document.createElement("div");
        toast.id = "ml-toast";
        document.body.appendChild(toast);
    }
    toast.className = `viz-toast ${type}`;
    const icon = type === "success" ? "fa-check-circle" : "fa-exclamation-circle";
    toast.innerHTML = `<i class="fas ${icon}"></i> ${message}`;
    requestAnimationFrame(() => toast.classList.add("show"));
    clearTimeout(toast._hideTimeout);
    toast._hideTimeout = setTimeout(() => toast.classList.remove("show"), 3500);
}

window.onload = initML;