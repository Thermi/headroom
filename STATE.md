# State

Tracking concurrent work on the project.

## Current: Fix tree-sitter version bounds for real

- **Status:** Applied
- **Who:** headroom-agent
- **What:** Fixed `tree-sitter` lower bound to `>=0.25.2` (all `tree-sitter-language-pack` 0.x versions require `>=0.25.2`). Previous incorrect upper bound `<0.25.0` made resolution impossible. Also fixed error message in `code_compressor.py` to match.
- **Files changed:** `pyproject.toml`, `headroom/transforms/code_compressor.py`, `STATE.md`
- **Commit:** TBD
