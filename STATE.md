# State

Tracking concurrent work on the project.

## Current: Lock tree-sitter version bounds to prevent AST runtime mismatch

- **Status:** Already applied by concurrent agent in `bc1cf2a7`
- **Who:** Noel Kuntze (concurrent agent)
- **What:** Added upper bounds (`<0.25.0`, `<0.12.0`) to `tree-sitter` and `tree-sitter-language-pack` in `pyproject.toml` to prevent AST runtime version mismatch. Also removed dead tree-sitter parser preload at startup.
- **Files changed:** `pyproject.toml`, `headroom/transforms/content_router.py`
- **Note:** My edit to `pyproject.toml` matched the concurrent commit — no further action needed.
