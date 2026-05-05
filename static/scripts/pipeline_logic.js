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

function updateStage(stageName, message, progress = 0) {
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
        const time = new Date().toLocaleTimeString([], { hour12: false });
        log.innerHTML = `<span style="color: #666;">[${time}]</span> <strong style="color: #a333ff;">${stageName}:</strong> ${message}`;
        consoleBox.appendChild(log);
        consoleBox.scrollTop = consoleBox.scrollHeight;
    }
}

/**
 * Start pipeline SSE stream.
 * If file is null, the backend uses the raw DF already stored in the session.
 */
async function startPipelineStream(file) {
    const formData = new FormData();
    // Only append file if provided (first-time upload without prior /process call)
    if (file) {
        formData.append("file", file);
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
    }
}

function resetPipeline() {
    const consoleBox = document.getElementById("pipeline-console");
    if (consoleBox) consoleBox.innerHTML = '<div class="log-entry">System Standby. Ready for injection.</div>';
    document.querySelectorAll(".stage-item").forEach(item => {
        item.classList.remove("active", "completed");
    });
}
