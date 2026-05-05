/* ============================================================
   BRight AI — Phase 2: Visualization Logic
   Uses preprocessed (clean) data already stored in session.
   Upload zone only shown if no session data exists.
   ============================================================ */

const Phase2 = {
    uploadedFile: null,
    columns: [],
    numCols: [],
    catCols: [],
    dataFromSession: false,   // true when loaded from session
};

/* ════════════════════════════════════════════
   DOM READY
════════════════════════════════════════════ */
document.addEventListener("DOMContentLoaded", () => {
    initUploadZone();
    initTabs();
    tryLoadFromSession();   // ← auto-load clean data if available
});


/* ════════════════════════════════════════════
   AUTO-LOAD FROM SESSION
════════════════════════════════════════════ */
async function tryLoadFromSession() {
    try {
        const res  = await fetch("/phase2/status");
        const data = await res.json();

        if (data.status === "ready") {
            // Clean data available — populate UI without asking for upload
            Phase2.dataFromSession = true;
            Phase2.columns = data.columns;
            Phase2.numCols = data.columns.filter(c => c.type === "num").map(c => c.name);
            Phase2.catCols = data.columns.filter(c => c.type === "cat").map(c => c.name);

            // Mark upload zone as "already loaded"
            const zone  = document.getElementById("p2-upload-zone");
            const label = document.getElementById("p2-file-name-label");
            if (zone) {
                zone.classList.add("loaded-from-session");
                zone.querySelector("h3").textContent = "✅ Preprocessed Data Loaded";
                zone.querySelector("p").textContent  =
                    `${data.rows} rows × ${data.cols} columns — clean data from pipeline`;
            }
            if (label) {
                label.style.display = "inline-block";
                label.textContent   = `📊 ${data.rows} rows · ${data.cols} columns (preprocessed)`;
            }

            renderColumnsPanel(data);
            populateAllSelects();
            showVizTabs();
            showToast(`Clean dataset loaded — ${data.rows} rows ready for visualization.`, "success");
        }
        // If no_data: upload zone stays visible, user must upload
    } catch (err) {
        // Silently ignore — upload zone stays for manual upload
    }
}


/* ════════════════════════════════════════════
   1. UPLOAD ZONE — Drag & Drop + Click
   (Only used if no session data available)
════════════════════════════════════════════ */
function initUploadZone() {
    const zone      = document.getElementById("p2-upload-zone");
    const fileInput = document.getElementById("p2-file-input");
    const fileLabel = document.getElementById("p2-file-name-label");

    if (!zone || !fileInput) return;

    zone.addEventListener("click", () => {
        // Don't trigger file picker if already loaded from session
        if (!Phase2.dataFromSession) fileInput.click();
    });

    zone.addEventListener("dragover", (e) => {
        e.preventDefault();
        if (!Phase2.dataFromSession) zone.classList.add("drag-over");
    });

    zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));

    zone.addEventListener("drop", (e) => {
        e.preventDefault();
        zone.classList.remove("drag-over");
        if (Phase2.dataFromSession) return;
        const file = e.dataTransfer.files[0];
        if (file) handleFileSelected(file);
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) handleFileSelected(fileInput.files[0]);
    });
}


/* ════════════════════════════════════════════
   FILE UPLOAD HANDLER (manual upload fallback)
════════════════════════════════════════════ */
async function handleFileSelected(file) {
    const allowed = ["csv", "xlsx", "xls"];
    const ext = file.name.split(".").pop().toLowerCase();
    if (!allowed.includes(ext)) {
        showToast("Unsupported format. Use CSV or Excel files.", "error");
        return;
    }

    Phase2.uploadedFile = file;
    Phase2.dataFromSession = false;

    const label = document.getElementById("p2-file-name-label");
    if (label) {
        label.style.display = "inline-block";
        label.textContent   = "📂 " + file.name;
    }

    showToast("Scanning & preprocessing columns...", "success");

    try {
        const formData = new FormData();
        formData.append("file", file);

        // detect-columns now runs full preprocessing on the file
        const res  = await fetch("/phase2/detect-columns", { method: "POST", body: formData });
        const data = await res.json();

        if (data.status !== "success") {
            showToast(data.message || "Column detection failed.", "error");
            return;
        }

        Phase2.columns = data.columns;
        Phase2.numCols = data.columns.filter(c => c.type === "num").map(c => c.name);
        Phase2.catCols = data.columns.filter(c => c.type === "cat").map(c => c.name);

        renderColumnsPanel(data);
        populateAllSelects();
        showVizTabs();
        showToast("Dataset preprocessed — " + data.rows + " clean rows ready.", "success");

    } catch (err) {
        showToast("Connection error: " + err.message, "error");
    }
}


/* ════════════════════════════════════════════
   2. COLUMNS PANEL
════════════════════════════════════════════ */
function renderColumnsPanel(data) {
    const panel     = document.getElementById("p2-columns-panel");
    const chipsWrap = document.getElementById("p2-columns-chips");
    const metaRows  = document.getElementById("p2-meta-rows");
    const metaCols  = document.getElementById("p2-meta-cols");

    if (!panel) return;

    if (metaRows) metaRows.textContent = data.rows + " rows";
    if (metaCols) metaCols.textContent = data.cols + " columns";

    if (chipsWrap) {
        chipsWrap.innerHTML = data.columns.map(col => `
            <div class="col-chip ${col.type === 'num' ? 'numeric' : 'categorical'}"
                 title="Type: ${col.type === 'num' ? 'Numeric' : 'Categorical'}">
                <i class="fas ${col.type === 'num' ? 'fa-hashtag' : 'fa-tag'}"></i>
                <span>${col.name}</span>
                <span class="col-type-badge">${col.type === 'num' ? 'NUM' : 'CAT'}</span>
            </div>
        `).join("");
    }

    panel.style.display = "block";
}


/* ════════════════════════════════════════════
   3. POPULATE SELECTS
════════════════════════════════════════════ */
function populateAllSelects() {
    const { numCols, catCols, columns } = Phase2;
    const allCols = columns.map(c => c.name);

    const makeOptions = (cols, placeholder = "— Select column —") => {
        const blank = `<option value="">` + placeholder + `</option>`;
        return blank + cols.map(c => `<option value="${c}">${c}</option>`).join("");
    };

    const fill = (id, cols, placeholder) => {
        const el = document.getElementById(id);
        if (el) el.innerHTML = makeOptions(cols, placeholder);
    };

    fill("scatter2d-x",     numCols, "— X Axis (Numeric) —");
    fill("scatter2d-y",     numCols, "— Y Axis (Numeric) —");
    fill("scatter2d-color", allCols, "— Color By (Optional) —");
    fill("scatter3d-x",     numCols, "— X Axis (Numeric) —");
    fill("scatter3d-y",     numCols, "— Y Axis (Numeric) —");
    fill("scatter3d-z",     numCols, "— Z Axis (Numeric) —");
    fill("scatter3d-color", allCols, "— Color By (Optional) —");
    fill("joint-col1",      numCols, "— Column 1 (Numeric) —");
    fill("joint-col2",      numCols, "— Column 2 (Numeric) —");
    fill("stacked-col1",    catCols.length ? catCols : allCols, "— Primary Column —");
    fill("stacked-col2",    catCols.length ? catCols : allCols, "— Secondary Column —");
    fill("crosstab-col1",   catCols.length ? catCols : allCols, "— Row Variable —");
    fill("crosstab-col2",   catCols.length ? catCols : allCols, "— Column Variable —");
    fill("violin-num",      numCols, "— Numeric Column —");
    fill("violin-cat",      catCols.length ? catCols : allCols, "— Category Column —");
    fillMultiSelect("facet-numcols", numCols);
    fill("facet-cat",       catCols.length ? catCols : allCols, "— Facet Category —");
    fill("bubble-x",        allCols, "— X Axis —");
    fill("bubble-y",        numCols, "— Y Axis (Numeric) —");
    fill("bubble-size",     numCols, "— Bubble Size (Numeric) —");
    fill("bubble-color",    allCols, "— Color By (Optional) —");
}

function fillMultiSelect(id, cols) {
    const el = document.getElementById(id);
    if (!el) return;
    el.innerHTML = cols.map(c => `<option value="${c}">${c}</option>`).join("");
}


/* ════════════════════════════════════════════
   4. TAB NAVIGATION
════════════════════════════════════════════ */
function initTabs() {
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const target = btn.dataset.tab;
            document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
            btn.classList.add("active");
            const panel = document.getElementById("tab-" + target);
            if (panel) panel.classList.add("active");
        });
    });
}

function showVizTabs() {
    const wrapper = document.getElementById("p2-viz-tabs");
    if (wrapper) {
        wrapper.style.display = "block";
        const firstBtn   = document.querySelector(".tab-btn");
        const firstPanel = document.querySelector(".tab-panel");
        if (firstBtn)   firstBtn.classList.add("active");
        if (firstPanel) firstPanel.classList.add("active");
    }
}


/* ════════════════════════════════════════════
   5. GENERATE CHART (session-based, no file re-upload)
════════════════════════════════════════════ */
async function generateChart(chartType, params, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    // Require either session data or a manually uploaded file
    if (!Phase2.dataFromSession && !Phase2.uploadedFile) {
        showToast("Please upload a file or run the pipeline first.", "error");
        return;
    }

    const chartLabel = chartType.replace(/_/g, " ").toUpperCase();

    const loadingHints = {
        summary_dashboard:   "Building overview dashboard...",
        correlation_heatmap: "Computing correlations...",
        scatter_2d:          "Plotting 2D scatter...",
        scatter_3d:          "Rendering 3D space...",
        joint_plot:          "Calculating joint distribution...",
        stacked_bar:         "Stacking categories...",
        cross_tabulation:    "Building cross-tab matrix...",
        violin_plot:         "Drawing violin distribution...",
        facet_grid:          "Generating facet grid panels...",
        bubble_chart:        "Inflating bubble chart...",
    };
    const hint = loadingHints[chartType] || "Generating chart...";

    container.innerHTML = `
        <div class="viz-result-card" style="min-height:220px;display:flex;align-items:center;justify-content:center;">
            <div style="text-align:center;padding:30px 20px;">
                <div class="viz-spinner" style="margin:0 auto 18px;"></div>
                <p style="font-size:11px;letter-spacing:2px;color:var(--phase2-accent);text-transform:uppercase;margin-bottom:6px;">${hint}</p>
                <p style="font-size:10px;letter-spacing:1px;color:var(--txt-dim);text-transform:uppercase;opacity:0.5;">${chartLabel}</p>
                <div class="viz-skeleton" style="height:8px;border-radius:4px;margin-top:20px;width:70%;margin-left:auto;margin-right:auto;"></div>
            </div>
        </div>
    `;

    const formData = new FormData();
    formData.append("chart_type", chartType);

    // Only append file if NOT using session data
    if (!Phase2.dataFromSession && Phase2.uploadedFile) {
        formData.append("file", Phase2.uploadedFile);
    }

    for (const [key, val] of Object.entries(params)) {
        if (val !== null && val !== undefined && val !== "") {
            if (Array.isArray(val)) {
                val.forEach(v => formData.append(key, v));
            } else {
                formData.append(key, val);
            }
        }
    }

    try {
        const res  = await fetch("/phase2/generate", { method: "POST", body: formData });
        const data = await res.json();

        if (data.status !== "success") {
            container.innerHTML = renderError(data.message || "Generation failed.");
            showToast(data.message || "Chart error.", "error");
            return;
        }

        renderResult(container, data);
        showToast("Chart generated successfully!", "success");

    } catch (err) {
        container.innerHTML = renderError("Server error: " + err.message);
        showToast("Request failed.", "error");
    }
}


/* ════════════════════════════════════════════
   RESULT RENDERING
════════════════════════════════════════════ */
function renderResult(container, data) {
    const viewUrl = data.view_url;
    const dlUrl   = data.download_url;
    const label   = data.chart_type.replace(/_/g, " ").toUpperCase();
    const isHtml  = data.plot_type === "html";

    const plotArea = isHtml
        ? `<iframe class="plot-iframe" src="${viewUrl}" loading="lazy"></iframe>`
        : `<img class="plot-preview-img" src="${viewUrl}?t=${Date.now()}" alt="${label}"
               onclick="openLightbox('${viewUrl}', '${label}', false)" title="Click to enlarge" />`;

    container.innerHTML = `
        <div class="viz-result-card">
            <div class="result-header">
                <div class="result-title">
                    <i class="fas fa-chart-bar"></i>
                    ${label}
                </div>
                <div class="result-actions">
                    <button class="expand-btn" onclick="openLightbox('${viewUrl}', '${label}', ${isHtml})">
                        <i class="fas fa-expand-arrows-alt"></i> Expand
                    </button>
                    <a href="${viewUrl}" target="_blank" class="result-action-btn view-btn">
                        <i class="fas fa-external-link-alt"></i> Tab
                    </a>
                    <a href="${dlUrl}" download class="result-action-btn download-btn">
                        <i class="fas fa-download"></i> Download
                    </a>
                </div>
            </div>
            ${plotArea}
            <div class="chart-name-label">${label}</div>
        </div>
    `;
}


/* ════════════════════════════════════════════
   LIGHTBOX
════════════════════════════════════════════ */
function openLightbox(url, label, isHtml) {
    const lb     = document.getElementById("chart-lightbox");
    const img    = document.getElementById("lightbox-img");
    const iframe = document.getElementById("lightbox-iframe");
    const title  = document.getElementById("lightbox-title");
    const openBtn= document.getElementById("lightbox-open");

    if (!lb) return;
    title.textContent = label;
    openBtn.href      = url;

    if (isHtml) {
        img.style.display    = "none";
        iframe.style.display = "block";
        iframe.src           = url;
    } else {
        iframe.style.display = "none";
        img.style.display    = "block";
        img.src              = url + "?t=" + Date.now();
    }

    lb.style.display    = "flex";
    lb.style.alignItems = "center";
    lb.style.justifyContent = "center";
    document.body.style.overflow = "hidden";
}

function closeLightbox() {
    const lb     = document.getElementById("chart-lightbox");
    const iframe = document.getElementById("lightbox-iframe");
    if (lb) lb.style.display = "none";
    if (iframe) iframe.src = "";
    document.body.style.overflow = "";
}

document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeLightbox();
});

function renderError(msg) {
    return `
        <div class="viz-result-card" style="border-color: rgba(255,64,129,0.3);">
            <p style="color: var(--phase2-red); text-align: center; padding: 30px; font-size: 13px;">
                <i class="fas fa-exclamation-triangle" style="margin-right: 8px;"></i>
                ${msg}
            </p>
        </div>
    `;
}


/* ════════════════════════════════════════════
   6. CHART HANDLERS
════════════════════════════════════════════ */
function generateSummaryDashboard()    { generateChart("summary_dashboard",   {}, "result-summary"); }
function generateCorrelationHeatmap()  { generateChart("correlation_heatmap", {}, "result-correlation"); }

function generateScatter2D() {
    const params = {
        col1:      document.getElementById("scatter2d-x")?.value,
        col2:      document.getElementById("scatter2d-y")?.value,
        color_col: document.getElementById("scatter2d-color")?.value,
    };
    if (!params.col1 || !params.col2) { showToast("Please select X and Y columns.", "error"); return; }
    generateChart("scatter_2d", params, "result-scatter2d");
}

function generateScatter3D() {
    const params = {
        col1:      document.getElementById("scatter3d-x")?.value,
        col2:      document.getElementById("scatter3d-y")?.value,
        col3:      document.getElementById("scatter3d-z")?.value,
        color_col: document.getElementById("scatter3d-color")?.value,
    };
    if (!params.col1 || !params.col2 || !params.col3) { showToast("Please select all three axes.", "error"); return; }
    generateChart("scatter_3d", params, "result-scatter3d");
}

function generateJointPlot() {
    const params = {
        col1: document.getElementById("joint-col1")?.value,
        col2: document.getElementById("joint-col2")?.value,
        kind: document.getElementById("joint-kind")?.value || "scatter",
    };
    if (!params.col1 || !params.col2) { showToast("Please select both columns.", "error"); return; }
    generateChart("joint_plot", params, "result-joint");
}

function generateStackedBar() {
    const params = {
        col1:      document.getElementById("stacked-col1")?.value,
        col2:      document.getElementById("stacked-col2")?.value,
        normalize: document.getElementById("stacked-normalize")?.checked ? "true" : "false",
    };
    if (!params.col1 || !params.col2) { showToast("Please select both columns.", "error"); return; }
    generateChart("stacked_bar", params, "result-stacked");
}

function generateCrossTab() {
    const params = {
        col1: document.getElementById("crosstab-col1")?.value,
        col2: document.getElementById("crosstab-col2")?.value,
    };
    if (!params.col1 || !params.col2) { showToast("Please select both columns.", "error"); return; }
    generateChart("cross_tabulation", params, "result-crosstab");
}

function generateViolinPlot() {
    const params = {
        num_col: document.getElementById("violin-num")?.value,
        cat_col: document.getElementById("violin-cat")?.value,
    };
    if (!params.num_col || !params.cat_col) { showToast("Please select both columns.", "error"); return; }
    generateChart("violin_plot", params, "result-violin");
}

function generateFacetGrid() {
    const multiSelect  = document.getElementById("facet-numcols");
    const selectedNums = multiSelect ? Array.from(multiSelect.selectedOptions).map(o => o.value) : [];
    const params = {
        num_cols: selectedNums,
        cat_col:  document.getElementById("facet-cat")?.value,
    };
    if (!selectedNums.length || !params.cat_col) { showToast("Please select at least one numeric column and a category.", "error"); return; }
    generateChart("facet_grid", params, "result-facet");
}

function generateBubbleChart() {
    const params = {
        x:     document.getElementById("bubble-x")?.value,
        y:     document.getElementById("bubble-y")?.value,
        size:  document.getElementById("bubble-size")?.value,
        color: document.getElementById("bubble-color")?.value,
    };
    if (!params.x || !params.y || !params.size) { showToast("Please select X, Y, and Size columns.", "error"); return; }
    generateChart("bubble_chart", params, "result-bubble");
}


/* ════════════════════════════════════════════
   7. TOAST
════════════════════════════════════════════ */
function showToast(message, type = "success") {
    let toast = document.getElementById("p2-toast");
    if (!toast) {
        toast = document.createElement("div");
        toast.id        = "p2-toast";
        toast.className = "viz-toast";
        document.body.appendChild(toast);
    }
    toast.className = `viz-toast ${type}`;
    const icon = type === "success" ? "fa-check-circle" : "fa-exclamation-circle";
    toast.innerHTML = `<i class="fas ${icon}"></i> ${message}`;
    requestAnimationFrame(() => toast.classList.add("show"));
    clearTimeout(toast._hideTimeout);
    toast._hideTimeout = setTimeout(() => toast.classList.remove("show"), 3500);
}
