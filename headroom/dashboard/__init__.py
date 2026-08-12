"""Headroom Dashboard - Real-time proxy monitoring UI."""

from pathlib import Path

DASHBOARD_DIR = Path(__file__).parent
TEMPLATES_DIR = DASHBOARD_DIR / "templates"
# Vendored tailwind/htmx/alpine. Served locally because Edge's Tracking
# Prevention and corporate proxies block unpkg.com/cdn.tailwindcss.com, which
# left the dashboard unstyled and dataless on some Windows machines.
STATIC_DIR = DASHBOARD_DIR / "static"


def get_dashboard_html() -> str:
    """Load the dashboard HTML template."""
    template_path = TEMPLATES_DIR / "dashboard.html"
    try:
        # ``newline`` preserves the asset's exact line endings on Python 3.13+.
        return template_path.read_text(encoding="utf-8", newline="")  # type: ignore[call-arg]
    except TypeError:
        # ``Path.read_text`` gained ``newline`` after some supported Python versions.
        with template_path.open("r", encoding="utf-8", newline="") as template:
            return template.read()


def get_settings_html() -> str:
    """Load the settings GUI HTML template."""
    template_path = TEMPLATES_DIR / "settings.html"
    return template_path.read_text(encoding="utf-8")
