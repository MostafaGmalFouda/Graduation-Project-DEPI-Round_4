/* ============================================================
   BRight AI — Phase 5: ML Logic
   ============================================================ */

const ML = { columns: [], guessedTarget: null, availableModels: {}, lastResult: null };

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

        document.getElementById("ml-workspace").style.display = "block";

        if (APP_MODE === "developer") {
            document.getElementById("ml-developer-mode").style.display = "block";
            initDeveloperForm(data);
        } else {
            document.getElementById("ml-user-mode").style.display = "block";
            document.getElementById("ml-meta-rows").textContent = `${data.rows} rows`;
            document.getElementById("ml-meta-cols").textContent = `${data.cols} columns`;

            const targetSelect = document.getElementById("ml-user-target");
            targetSelect.innerHTML = data.columns.map(c =>
                `<option value="${c}" ${c === data.guessed_target ? "selected" : ""}>${c}</option>`
            ).join("");
            document.getElementById("ml-guessed-badge").style.display = "inline-block";
        }
    } catch (e) {
        showToast("Could not reach the server: " + e.message, "error");
    }
}

function initDeveloperForm(data) {
    const targetSelect = document.getElementById("ml-dev-target");
    targetSelect.innerHTML = data.columns.map(c =>
        `<option value="${c}" ${c === data.guessed_target ? "selected" : ""}>${c}</option>`
    ).join("");

    const taskSelect = document.getElementById("ml-dev-task");
    const modelSelect = document.getElementById("ml-dev-model");

    function refreshModelOptions() {
        const task = taskSelect.value || data.guessed_task_type;
        const models = ML.availableModels[task] || ML.availableModels.classification;
        modelSelect.innerHTML = models.map(m =>
            `<option value="${m}">${m.replace(/_/g, " ")}</option>`
        ).join("");
    }
    taskSelect.addEventListener("change", refreshModelOptions);
    refreshModelOptions();

    function refreshFeatureChecklist() {
        const target = targetSelect.value;
        const container = document.getElementById("ml-dev-features");
        container.innerHTML = data.columns.filter(c => c !== target).map(c => `
            <label class="col-chip categorical" style="cursor:pointer;">
                <input type="checkbox" class="ml-feature-check" value="${c}" checked style="margin-right:6px;">${c}
            </label>
        `).join("");
    }
    targetSelect.addEventListener("change", refreshFeatureChecklist);
    refreshFeatureChecklist();
}

document.getElementById("ml-auto-run-btn")?.addEventListener("click", () => trainML(true));
document.getElementById("ml-dev-run-btn")?.addEventListener("click", () => trainML(false));

async function trainML(auto) {
    const loading = document.getElementById("ml-loading");
    const results = document.getElementById("ml-results");
    loading.classList.add("active");
    results.style.display = "none";

    const formData = new FormData();
    formData.append("mode", auto ? "user" : "developer");

    if (auto) {
        const target = document.getElementById("ml-user-target").value;
        if (target) formData.append("target_column", target);
    } else {
        formData.append("target_column", document.getElementById("ml-dev-target").value);
        formData.append("model_name", document.getElementById("ml-dev-model").value);
        const task = document.getElementById("ml-dev-task").value;
        if (task) formData.append("task_type", task);
        formData.append("test_size", document.getElementById("ml-dev-testsize").value);
        formData.append("scale", document.getElementById("ml-dev-scale").checked ? "true" : "false");

        const nEst = document.getElementById("ml-dev-n-estimators").value;
        if (nEst) formData.append("n_estimators", nEst);
        const maxDepth = document.getElementById("ml-dev-max-depth").value;
        if (maxDepth) formData.append("max_depth", maxDepth);

        document.querySelectorAll(".ml-feature-check:checked").forEach(cb => {
            formData.append("feature_columns", cb.value);
        });
    }

    try {
        const res = await fetch("/ml/train", { method: "POST", body: formData });
        const data = await res.json();
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
        loading.classList.remove("active");
        showToast("Request failed: " + e.message, "error");
    }
}

function renderMLResults(data) {
    const metrics = data.metrics;
    const cards = document.getElementById("ml-metric-cards");

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

    // Build the "try a prediction" form from the model's feature columns
    const predictForm = document.getElementById("ml-predict-form");
    predictForm.innerHTML = data.feature_columns.map(f => `
        <div class="chart-config-card" style="padding:12px 16px;">
            <div class="form-group" style="margin-bottom:0;">
                <label>${f}</label>
                <input type="number" step="any" class="form-select ml-predict-input" data-feature="${f}" placeholder="value">
            </div>
        </div>
    `).join("");
}

document.getElementById("ml-predict-btn")?.addEventListener("click", async () => {
    if (!ML.lastResult) return;
    const row = {};
    document.querySelectorAll(".ml-predict-input").forEach(inp => {
        row[inp.dataset.feature] = inp.value || 0;
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
