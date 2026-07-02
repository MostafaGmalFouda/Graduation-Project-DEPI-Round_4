/* =====================================================================
   BRight AI — Shared Sidebar Behavior
   Collapse (desktop) + mobile toggle only. Used on every phase page
   (Phase 2 / NLP / ML) that includes the shared sidebar partial but not
   the full app_shell.js (which is index.html / Phase 1 specific).
===================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // ── Sidebar collapse (desktop) ──────────────────────────────────────
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

    // ── Mobile sidebar toggle (hamburger buttons) ───────────────────────
    function toggleMobileSidebar() {
        const sidebar = document.getElementById('sidebar');
        if (sidebar) sidebar.classList.toggle('mobile-open');
    }
    document.querySelectorAll('.mobile-sidebar-toggle').forEach(btn => {
        btn.onclick = toggleMobileSidebar;
    });
});
