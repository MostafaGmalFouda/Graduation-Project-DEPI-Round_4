const stagesOrder = ["Data Validation", "Preprocessing", "Outlier Detection", "Report Generated"];

function renderPipeline() {
    const container = document.getElementById("pipeline-stages-container");
    if (!container) return;
    container.innerHTML = stagesOrder.map((stage, index) => `
        <div class="stage-item" data-stage="${stage}">
            <div class="stage-icon"><i class="fas fa-microchip"></i></div>
            <div class="stage-info">
                <h4>${stage}</h4>
                <p>Waiting for engine...</p>
            </div>
            <div class="stage-status"><i class="fas fa-clock"></i></div>
        </div>
        ${index < stagesOrder.length - 1 ? '<div class="pipeline-line"></div>' : ''}
    `).join("");
}

function updateStage(stageName, message, progress = 0, historicalTime = null) {
    document.querySelectorAll(".stage-item").forEach(item => {
        const name = item.dataset.stage;
        if (name === stageName) {
            item.classList.add("active");
            item.querySelector(".stage-status i").className = "fas fa-sync fa-spin";
            item.querySelector(".stage-info p").innerText = message;
            if (stageName === stagesOrder[stagesOrder.length - 1] && progress === 100) {
                setTimeout(() => {
                    item.classList.add("completed");
                    item.querySelector(".stage-status i").className = "fas fa-check-double";
                    item.querySelector(".stage-info p").innerText = "Verified ✓";
                }, 300);
            }
        } else if (stagesOrder.indexOf(name) < stagesOrder.indexOf(stageName)) {
            setTimeout(() => {
                item.classList.add("completed");
                item.querySelector(".stage-status i").className = "fas fa-check-double";
                item.querySelector(".stage-info p").innerText = "Verified ✓";
            }, 90);
        }
    });

    const consoleBox = document.getElementById("pipeline-console");
    if (consoleBox) {
        const log = document.createElement("div");
        log.className = "log-entry active-log";
        // Replaying a finished run passes back its ORIGINAL server-recorded
        // time; a live run has no historicalTime and just uses "now".
        const time = historicalTime || new Date().toLocaleTimeString([], { hour12: false });
        log.innerHTML = `<span style="color: #666;">[${time}]</span> <strong style="color: #a333ff;">${stageName}:</strong> ${message}`;
        consoleBox.appendChild(log);
        consoleBox.scrollTop = consoleBox.scrollHeight;
    }
}

/**
 * Start pipeline SSE stream.
 * If file is null, the backend uses the raw DF already stored in the session.
 */
async function startPipelineStream(file, devParams = {}) {
    const formData = new FormData();
    // Only append file if provided (first-time upload without prior /process call)
    if (file) {
        formData.append("file", file);
    }
      if (devParams && typeof devParams === 'object') {
        Object.entries(devParams).forEach(([key, value]) => {
            formData.append(key, value);
        });
    }
    
    try {
        const res = await fetch("/pipeline-stream", { method: "POST", body: formData });
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            let parts = buffer.split("\n\n");
            buffer = parts.pop();

            for (let p of parts) {
                if (!p.trim()) continue;
                const json = JSON.parse(p.replace("data:", "").trim());

                if (json.stage) {
                    updateStage(json.stage, json.message, json.progress);
                }

                if (json.done) {
                    updateStage("Report Generated", "Final Intelligence Deployed", 100);
                    // NOTE: we intentionally do NOT clear the chat here.
                    // Running the pipeline processes the SAME uploaded
                    // dataset (it's not a new upload) — the chat should stay
                    // exactly as it was, and now the assistant can ALSO
                    // answer questions about the clean data alongside the
                    // raw data it already knew about. The chat is only
                    // cleared on a real page refresh or a brand new upload.
                    if (typeof handleFinalReport === "function") {
                        handleFinalReport(json);
                    }
                    return;
                }
            }
        }
    } catch (err) {
        console.error("Stream Error:", err);
        const streamWrapper = document.querySelector('.stream-wrapper');
        if (streamWrapper) streamWrapper.classList.remove('loading');

        // Previously this only cleared the spinner class and stopped —
        // the "ENGINE RUNNING..." button and the "System Standby" console
        // line were left exactly as they were, so on any real failure
        // (network drop, server 500 before streaming even started, etc.)
        // the UI looked like it was still working forever with nothing
        // ever appearing. Surface it instead.
        const consoleBox = document.getElementById("pipeline-console");
        if (consoleBox) {
            const log = document.createElement("div");
            log.className = "log-entry active-log";
            log.style.color = "#ff4d6d";
            log.innerHTML = `<span style="color:#666;">[${new Date().toLocaleTimeString([], { hour12: false })}]</span> <strong style="color:#ff4d6d;">Error:</strong> Lost connection to the engine (${err.message}). Check the server console for a traceback, then try again.`;
            consoleBox.appendChild(log);
            consoleBox.scrollTop = consoleBox.scrollHeight;
        }

        // These ids are declared on the main pipeline page (index.html) —
        // guard with typeof since pipeline_logic.js is shared with other
        // pages that don't have a stream button.
        if (typeof streamBtn !== "undefined" && streamBtn) {
            streamBtn.disabled = false;
            streamBtn.innerText = "RETRY ENGINE";
            streamBtn.style.background = "#ff4d6d";
        }
        if (typeof errorBox !== "undefined" && errorBox) {
            errorBox.innerText = "Pipeline failed: " + err.message + ". Check the server console for details, then retry.";
            errorBox.style.display = "block";
        }
    }
}

function resetPipeline() {
    const consoleBox = document.getElementById("pipeline-console");
    if (consoleBox) consoleBox.innerHTML = '<div class="log-entry">System Standby. Ready for injection.</div>';
    document.querySelectorAll(".stage-item").forEach(item => {
        item.classList.remove("active", "completed");
    });
}