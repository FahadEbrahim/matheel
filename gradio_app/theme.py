"""Styles and browser layout helpers for the Gradio workspace."""

APP_CSS = """
.gradio-container {
    width: 100%;
    min-width: 0;
    max-width: 1480px;
    margin: 0 auto;
    padding: 0 18px 48px;
    box-sizing: border-box;
    overflow-x: clip;
}

.matheel-hero {
    position: relative;
    overflow: hidden;
    margin: 6px 0 18px;
    padding: clamp(24px, 4vw, 42px);
    border: 1px solid rgba(148, 210, 196, 0.24);
    border-radius: 22px;
    background:
        radial-gradient(circle at 92% 10%, rgba(45, 212, 191, 0.22), transparent 34%),
        linear-gradient(135deg, #0f172a 0%, #123a42 58%, #0f4c4b 100%);
    color: #f8fafc;
    box-shadow: 0 22px 54px rgba(15, 23, 42, 0.18);
}

.matheel-hero-topline,
.matheel-workflow-kicker {
    margin: 0 0 10px;
    color: #99f6e4;
    font-size: 0.76rem;
    font-weight: 750;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.matheel-hero h1 {
    margin: 0;
    color: #ffffff;
    font-size: clamp(2rem, 5vw, 3.75rem);
    line-height: 1;
    letter-spacing: -0.045em;
}

.matheel-hero-copy {
    max-width: 760px;
    margin: 14px 0 22px;
    color: #dbeafe;
    font-size: clamp(1rem, 2vw, 1.16rem);
    line-height: 1.65;
}

.matheel-journey {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
}

.matheel-journey-step {
    min-width: 0;
    padding: 12px 14px;
    border: 1px solid rgba(226, 232, 240, 0.14);
    border-radius: 14px;
    background: rgba(15, 23, 42, 0.28);
    color: #e2e8f0;
    line-height: 1.35;
}

.matheel-journey-step span {
    display: block;
    margin-bottom: 4px;
    color: #5eead4;
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

#matheel-workflows {
    gap: 14px;
    min-width: 0;
}

#matheel-workflows > .tab-wrapper {
    position: sticky;
    top: 8px;
    z-index: 20;
}

#matheel-workflows > .tab-wrapper > .tab-container[role="tablist"] {
    max-width: 100%;
    padding: 6px;
    border: 1px solid var(--border-color-primary);
    border-radius: 14px;
    background: color-mix(in srgb, var(--background-fill-primary) 92%, transparent);
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.1);
    backdrop-filter: blur(14px);
}

#matheel-workflows > .tab-wrapper > .tab-container[role="tablist"] button {
    min-height: 42px;
    border-radius: 10px;
    font-weight: 700;
}

.matheel-workflow-intro {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(220px, 0.42fr);
    gap: 18px;
    align-items: center;
    margin: 4px 0 16px;
    padding: 18px 20px;
    border: 1px solid var(--border-color-primary);
    border-radius: 16px;
    background: linear-gradient(135deg, var(--block-background-fill), var(--background-fill-secondary));
}

.matheel-workflow-intro h2 {
    margin: 0 0 6px;
    font-size: clamp(1.2rem, 2.5vw, 1.55rem);
    letter-spacing: -0.02em;
}

.matheel-workflow-intro p {
    margin: 0;
    color: var(--body-text-color-subdued);
    line-height: 1.55;
}

.matheel-workflow-kicker {
    color: var(--color-accent, #0f766e);
}

.matheel-workflow-outcome {
    padding: 12px 14px;
    border-left: 3px solid var(--color-accent, #0f766e);
    border-radius: 10px;
    background: var(--background-fill-primary);
    color: var(--body-text-color);
    font-size: 0.9rem;
    line-height: 1.5;
}

.matheel-workflow-outcome strong {
    display: block;
    margin-bottom: 2px;
}

.matheel-workflow-grid {
    align-items: flex-start;
    gap: 18px;
}

#dataset-setup-column {
    order: 1;
}

#dataset-results-column {
    order: 2;
}

.matheel-results-panel,
.matheel-control-panel {
    min-width: 0;
    max-width: 100%;
}

.matheel-control-panel {
    padding: 14px;
    border: 1px solid var(--border-color-primary);
    border-radius: 16px;
    background: var(--block-background-fill);
    box-shadow: 0 12px 32px rgba(15, 23, 42, 0.07);
}

.matheel-control-panel > .form,
.matheel-control-panel > .block {
    margin-bottom: 10px;
}

.matheel-control-panel .wrap {
    border-radius: 12px;
}

.matheel-subtabs > .tab-wrapper > .tab-container {
    margin-bottom: 14px;
    padding: 4px;
    border-radius: 12px;
    background: var(--background-fill-secondary);
}

.matheel-summary,
.matheel-status {
    border: 1px solid #d6d9df;
    border-left: 4px solid #2a7f62;
    border-radius: 8px;
    background: #fbfbf8;
    padding: 10px 12px;
    margin: 0 0 12px;
}

.matheel-status {
    background: #f6faf7;
}

.matheel-empty {
    border-left-color: #8a6f24;
    background: #fffaf0;
}

.matheel-error {
    border-left-color: #b42318;
    background: #fff5f5;
}

.matheel-summary-title {
    display: block;
    margin-bottom: 8px;
    color: #1f2328;
}

.matheel-summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 8px 12px;
}

.matheel-summary-item {
    min-width: 0;
}

.matheel-summary-label {
    display: block;
    color: #5f6773;
    font-size: 0.78rem;
    line-height: 1.2;
}

.matheel-summary-value {
    display: block;
    color: #1f2328;
    font-weight: 600;
    overflow-wrap: anywhere;
}

.matheel-table {
    font-size: 0.92rem;
}

button.primary {
    box-shadow: 0 8px 20px rgba(13, 148, 136, 0.18);
}

@media (min-width: 980px) {
    .matheel-control-panel {
        position: sticky;
        top: 76px;
        max-height: calc(100vh - 92px);
        overflow-y: auto;
        overscroll-behavior: contain;
    }
}

@media (max-width: 860px) {
    .gradio-container {
        padding: 0 8px 32px;
    }

    .matheel-journey {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .matheel-workflow-intro {
        grid-template-columns: 1fr;
    }

    #matheel-workflows > .tab-wrapper {
        position: static;
        height: auto !important;
        align-items: stretch !important;
    }

    #matheel-workflows > .tab-wrapper > .tab-container[role="tablist"] {
        position: static !important;
        display: grid !important;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 4px;
        height: auto !important;
        overflow: visible !important;
    }

    #matheel-workflows > .tab-wrapper > .tab-container[role="tablist"] button {
        width: 100% !important;
        min-width: 0;
        justify-content: center;
    }

    .matheel-workflow-grid {
        flex-direction: column;
        flex-wrap: nowrap;
    }

    .matheel-workflow-grid > .column {
        width: 100%;
        min-width: 0;
        flex: 1 1 auto;
    }
}

@media (max-width: 520px) {
    .matheel-hero {
        padding: 22px 18px;
        border-radius: 16px;
    }

    .matheel-journey {
        grid-template-columns: 1fr;
    }

    #matheel-workflows > .tab-wrapper > .tab-container[role="tablist"] {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

}
"""


APP_JS = """
() => {
    let animationFrame = null;

    const syncOverflowMenus = () => {
        document.querySelectorAll(".overflow-menu").forEach((overflowMenu) => {
            const trigger = overflowMenu.querySelector(":scope > button");
            const adjacentDropdown = trigger?.nextElementSibling;
            const dropdown =
                overflowMenu.querySelector(":scope > .overflow-dropdown") ||
                (adjacentDropdown?.classList.contains("overflow-dropdown")
                    ? adjacentDropdown
                    : overflowMenu.parentElement?.querySelector(".overflow-dropdown") ||
                      overflowMenu.closest(".tab-wrapper")?.querySelector(".overflow-dropdown"));
            if (!trigger) {
                return;
            }

            trigger.setAttribute("aria-label", "More workflow tabs");
            trigger.setAttribute("aria-haspopup", "menu");
            trigger.setAttribute(
                "aria-expanded",
                String(Boolean(dropdown && !dropdown.classList.contains("hide"))),
            );

            if (dropdown) {
                dropdown.setAttribute("role", "menu");
                dropdown.querySelectorAll("button").forEach((item) => {
                    item.setAttribute("role", "menuitem");
                });
            }
        });
    };

    const scheduleSync = () => {
        if (animationFrame !== null) {
            window.cancelAnimationFrame(animationFrame);
        }
        animationFrame = window.requestAnimationFrame(syncOverflowMenus);
    };

    syncOverflowMenus();
    const observer = new MutationObserver(scheduleSync);
    observer.observe(document.body, {
        subtree: true,
        childList: true,
        attributes: true,
        attributeFilter: ["class", "hidden", "aria-hidden"],
    });
    window.addEventListener("resize", scheduleSync);
}
"""
