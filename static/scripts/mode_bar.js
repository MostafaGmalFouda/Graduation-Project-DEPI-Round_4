/* =====================================================================
   BRight AI — Top Mode Bar (User Mode / Developer Mode toggle)
   Used on every phase page (Phase 2 / NLP / ML) that renders the shared
   components/mode_bar.html partial but does NOT load the full
   app_shell.js (that file is index.html / Phase 1 specific).

   Switches mode INSTANTLY, exactly like index.html does — no page
   reload/navigation of any kind. Each phase page's own logic file
   (ml_logic.js / nlp_logic.js / phase2_logic.js) exposes
   `window.applyPageMode(mode)`, which just flips which panel is visible
   (both panels' controls are already fully set up regardless of mode),
   so nothing the person uploaded or configured is ever lost.
===================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    const modeSwitch = document.getElementById('mode-switch');
    const labelUser = document.getElementById('label-user-side');
    const labelDev = document.getElementById('label-dev-side');
    const pill = document.getElementById('current-mode-pill');
    const pillText = document.getElementById('current-mode-text');
    if (!modeSwitch) return; // mode_bar.html partial not present on this page

    let currentMode = modeSwitch.classList.contains('dev-active') ? 'developer' : 'user';

    function applyPillUI(mode) {
        if (mode === 'developer') {
            modeSwitch.classList.add('dev-active');
            pill.classList.remove('user-pill');
            pill.classList.add('dev-pill');
            pill.querySelector('i').className = 'fas fa-code';
            pillText.innerText = 'Developer Mode';
            labelUser.classList.remove('is-active');
            labelDev.classList.add('is-active');
        } else {
            modeSwitch.classList.remove('dev-active');
            pill.classList.remove('dev-pill');
            pill.classList.add('user-pill');
            pill.querySelector('i').className = 'fas fa-bolt';
            pillText.innerText = 'User Mode';
            labelDev.classList.remove('is-active');
            labelUser.classList.add('is-active');
        }
    }

    async function switchTo(newMode) {
        if (newMode === currentMode) return;
        currentMode = newMode;

        // Instant visual feedback + instant panel swap — no reload, no
        // navigation, so uploaded data and in-progress form state stay put.
        applyPillUI(newMode);
        if (typeof window.applyPageMode === 'function') {
            window.applyPageMode(newMode);
        }

        // Persist for next time the page is server-rendered from scratch
        // (a real refresh, or visiting another phase page).
        try {
            await fetch('/set-mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: newMode })
            });
        } catch (e) {
            // Non-fatal: UI already reflects the switch; it'll just be
            // re-asked for on the next fresh page load if this failed.
        }
    }

    modeSwitch.onclick = () => switchTo(currentMode === 'developer' ? 'user' : 'developer');
    labelUser.onclick = () => switchTo('user');
    labelDev.onclick = () => switchTo('developer');
});