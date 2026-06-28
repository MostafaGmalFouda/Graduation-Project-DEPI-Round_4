/* =====================================================================
   BRight AI — App Shell Logic
   Sidebar navigation, Mode Select/Toggle, Developer Controls Panel
===================================================================== */

// ── Mode state ──────────────────────────────────────────────────────────
let currentMode = 'user'; // 'user' | 'developer'

function enterApp(mode) {
    currentMode = mode;
    const modeSelect = document.getElementById('mode-select-screen');
    const mainApp    = document.getElementById('main-app');

    modeSelect.style.opacity = '0';
    setTimeout(() => {
        modeSelect.style.display = 'none';
        mainApp.style.display = 'block';
        requestAnimationFrame(() => {
            mainApp.style.transition = 'opacity 0.6s ease';
            mainApp.style.opacity = '1';
        });
    }, 500);

    applyModeUI(mode);
}

function setBodyModeClass(mode) {
    document.body.classList.remove('mode-user', 'mode-developer');
    document.body.classList.add(mode === 'developer' ? 'mode-developer' : 'mode-user');
}

function applyModeUI(mode) {
    const devPanel   = document.getElementById('dev-controls-panel');
    const modeSwitch = document.getElementById('mode-switch');
    const pill       = document.getElementById('current-mode-pill');
    const pillText   = document.getElementById('current-mode-text');
    const labelUser  = document.getElementById('label-user-side');
    const labelDev   = document.getElementById('label-dev-side');

    setBodyModeClass(mode);

    if (mode === 'developer') {
        devPanel.style.display = 'block';
        modeSwitch.classList.add('dev-active');
        pill.classList.remove('user-pill');
        pill.classList.add('dev-pill');
        pill.querySelector('i').className = 'fas fa-code';
        pillText.innerText = 'Developer Mode';
        labelUser.classList.remove('is-active');
        labelDev.classList.add('is-active');
    } else {
        devPanel.style.display = 'none';
        modeSwitch.classList.remove('dev-active');
        pill.classList.remove('dev-pill');
        pill.classList.add('user-pill');
        pill.querySelector('i').className = 'fas fa-bolt';
        pillText.innerText = 'User Mode';
        labelDev.classList.remove('is-active');
        labelUser.classList.add('is-active');
    }
}

document.getElementById('select-user-mode').onclick = () => enterApp('user');
document.getElementById('select-dev-mode').onclick  = () => enterApp('developer');

async function saveMode(mode) {
    await fetch("/set-mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: mode })
    });
}

document.getElementById('mode-switch').onclick = async () => {
    currentMode = currentMode === 'user' ? 'developer' : 'user';
    applyModeUI(currentMode);
    await saveMode(currentMode);
};

document.getElementById('label-user-side').onclick = async () => {
    currentMode = 'user';
    applyModeUI('user');
    await saveMode('user');
};

document.getElementById('label-dev-side').onclick = async () => {
    currentMode = 'developer';
    applyModeUI('developer');
    await saveMode('developer');
};

// ── Generic pill-group helper ───────────────────────────────────────────
function setupPillGroup(groupId, hiddenInputId, onSelect) {
    const group = document.getElementById(groupId);
    const hiddenInput = document.getElementById(hiddenInputId);
    if (!group) return;
    group.querySelectorAll('.dev-pill-option').forEach(opt => {
        opt.onclick = () => {
            group.querySelectorAll('.dev-pill-option').forEach(o => o.classList.remove('selected'));
            opt.classList.add('selected');
            hiddenInput.value = opt.dataset.value;
            if (onSelect) onSelect(opt.dataset.value);
        };
    });
}

// ── Generic toggle-switch helper ────────────────────────────────────────
function setupToggleSwitch(switchId, hiddenInputId) {
    const sw = document.getElementById(switchId);
    const hiddenInput = document.getElementById(hiddenInputId);
    if (!sw) return;
    sw.onclick = () => {
        const isOn = sw.classList.toggle('on');
        hiddenInput.value = isOn ? 'true' : 'false';
    };
}

// ── 1. Null Drop Threshold slider ───────────────────────────────────────
const nullSlider     = document.getElementById('null-threshold-slider');
const nullValueLabel = document.getElementById('null-threshold-value');
if (nullSlider) {
    nullSlider.oninput = () => {
        nullValueLabel.innerText = Math.round(parseFloat(nullSlider.value) * 100) + '%';
    };
}

// ── 2. Null Fill Strategy ───────────────────────────────────────────────
setupPillGroup('null-fill-strategy-group', 'null-fill-strategy-input');

// ── 3. Outlier Detection Method (toggles Z-score threshold field) ──────
const zscoreField = document.getElementById('zscore-threshold-field');
setupPillGroup('outlier-method-group', 'outlier-method-input', (value) => {
    if (zscoreField) zscoreField.style.display = (value === 'zscore') ? 'block' : 'none';
});

// ── 4. Z-Score Threshold slider ─────────────────────────────────────────
const zscoreSlider = document.getElementById('zscore-threshold-slider');
const zscoreValueLabel = document.getElementById('zscore-threshold-value');
if (zscoreSlider) {
    zscoreSlider.oninput = () => {
        zscoreValueLabel.innerText = parseFloat(zscoreSlider.value).toFixed(1);
    };
}

// ── 5. Outlier Strategy ─────────────────────────────────────────────────
setupPillGroup('outlier-strategy-group', 'outlier-strategy-input');

// ── 6. Smart Type Conversion toggle ─────────────────────────────────────
setupToggleSwitch('type-conversion-switch', 'type-conversion-input');

// ── 7. Remove Duplicates toggle ─────────────────────────────────────────
setupToggleSwitch('remove-duplicates-switch', 'remove-duplicates-input');

// ── 8. Exclude Columns (chip picker) ────────────────────────────────────
let excludedColumnsSet = new Set();

function populateExcludeColumnsUI(columns) {
    const wrap = document.getElementById('exclude-columns-wrap');
    if (!wrap) return;
    excludedColumnsSet = new Set();
    wrap.innerHTML = '';
    wrap.classList.remove('empty');

    columns.forEach(col => {
        const chip = document.createElement('div');
        chip.className = 'dev-col-chip';
        chip.textContent = col;
        chip.dataset.col = col;
        chip.onclick = () => {
            if (excludedColumnsSet.has(col)) {
                excludedColumnsSet.delete(col);
                chip.classList.remove('excluded');
            } else {
                excludedColumnsSet.add(col);
                chip.classList.add('excluded');
            }
            syncExcludeColumnsInput();
        };
        wrap.appendChild(chip);
    });
    syncExcludeColumnsInput();
}

function clearExcludeColumnsUI() {
    const wrap = document.getElementById('exclude-columns-wrap');
    if (!wrap) return;
    excludedColumnsSet = new Set();
    wrap.innerHTML = '';
    wrap.classList.add('empty');
    syncExcludeColumnsInput();
}

function syncExcludeColumnsInput() {
    const hiddenInput = document.getElementById('exclude-columns-input');
    const countLabel = document.getElementById('exclude-count-value');
    if (hiddenInput) hiddenInput.value = Array.from(excludedColumnsSet).join(',');
    if (countLabel) countLabel.textContent = excludedColumnsSet.size + ' selected';
}

// ── 9. Categorical Encoding ─────────────────────────────────────────────
setupPillGroup('encoding-method-group', 'encoding-method-input');

// ── Collect all Developer Mode params for the pipeline stream ──────────
function getDevPipelineParams() {
    if (currentMode !== 'developer') return null;
    return {
        null_threshold: nullSlider ? nullSlider.value : '0.4',
        null_fill_strategy: document.getElementById('null-fill-strategy-input').value,
        outlier_method: document.getElementById('outlier-method-input').value,
        zscore_threshold: zscoreSlider ? zscoreSlider.value : '3.0',
        outlier_strategy: document.getElementById('outlier-strategy-input').value,
        do_type_conversion: document.getElementById('type-conversion-input').value,
        do_remove_duplicates: document.getElementById('remove-duplicates-input').value,
        exclude_columns: document.getElementById('exclude-columns-input').value,
        encoding_method: document.getElementById('encoding-method-input').value,
    };
}


// ═══════════════════════════════════════════════════════════════════════
// SIDEBAR NAVIGATION
// ═══════════════════════════════════════════════════════════════════════

const MODULE_INFO = {
    nlp: {
        icon: 'fa-comment-dots',
        title: 'NLP Module',
        text: 'Natural Language Processing tools — text classification, sentiment analysis, and entity extraction — are currently in development and will land in a future release of BRight AI.',
    },
    cv: {
        icon: 'fa-eye',
        title: 'Computer Vision Module',
        text: 'Image classification, object detection, and visual analytics tools are planned for an upcoming release of BRight AI.',
    },
    ml: {
        icon: 'fa-brain',
        title: 'AutoML Module',
        text: 'Automated model selection, training, and tuning across multiple algorithms — coming soon to BRight AI.',
    },
};

function showSection(target) {
    const modeSelect   = document.getElementById('mode-select-screen');
    const mainApp      = document.getElementById('main-app');
    const comingSoon   = document.getElementById('coming-soon-view');

    document.querySelectorAll('.sidebar-item').forEach(item => item.classList.remove('active'));

    if (target === 'eda') {
        document.getElementById('nav-eda').classList.add('active');
        comingSoon.style.display = 'none';
        // Show whichever EDA screen was last active (hero or workspace)
        if (mainApp.style.display === 'block') {
            mainApp.style.display = 'block';
            mainApp.style.opacity = '1';
        } else {
            modeSelect.style.display = 'block';
            modeSelect.style.opacity = '1';
        }
        return;
    }

    // Coming-soon modules (nlp / cv / ml)
    modeSelect.style.display = 'none';
    mainApp.style.display = 'none';
    comingSoon.style.display = 'block';

    const info = MODULE_INFO[target];
    if (info) {
        document.getElementById('coming-soon-icon-i').className = 'fas ' + info.icon;
        document.getElementById('coming-soon-title').innerText = info.title;
        document.getElementById('coming-soon-text').innerText = info.text;
    }

    const navItem = document.getElementById('nav-' + target);
    if (navItem) navItem.classList.add('active');

    // Close mobile sidebar after navigating
    document.getElementById('sidebar').classList.remove('mobile-open');
}

document.querySelectorAll('.sidebar-item').forEach(item => {
    item.addEventListener('click', () => {
        const target = item.dataset.target;
        if (!target) return;
        showSection(target);
    });
});

// ── Sidebar collapse (desktop) ──────────────────────────────────────────
const collapseBtn = document.getElementById('sidebar-collapse-btn');
if (collapseBtn) {
    collapseBtn.title = 'Collapse sidebar';
    collapseBtn.onclick = () => {
        const sidebar = document.getElementById('sidebar');
        sidebar.classList.toggle('collapsed');
        const icon = collapseBtn.querySelector('i');
        if (sidebar.classList.contains('collapsed')) {
            icon.className = 'fas fa-angles-right';
            collapseBtn.title = 'Expand sidebar';
        } else {
            icon.className = 'fas fa-angles-left';
            collapseBtn.title = 'Collapse sidebar';
        }
    };
}

// ── Mobile sidebar toggle (hamburger buttons) ───────────────────────────
function toggleMobileSidebar() {
    document.getElementById('sidebar').classList.toggle('mobile-open');
}
const mobileToggleHero = document.getElementById('mobile-sidebar-toggle-hero');
const mobileToggleApp  = document.getElementById('mobile-sidebar-toggle-app');
if (mobileToggleHero) mobileToggleHero.onclick = toggleMobileSidebar;
if (mobileToggleApp)  mobileToggleApp.onclick  = toggleMobileSidebar;
