# ONNX Artifact Recovery

## Goal

Recover from a malformed or stale Kompress ONNX artifact when the configured
artifact is a replaceable local file, without modifying bind-mounted or
protected files and without changing the existing candidate fallback behavior.

## Design

The ONNX candidate loader remains responsible for selecting an artifact and
running the construction plus smoke test. If either operation fails, the loader
may make one recovery attempt for that same artifact when all of these hold:

- Network downloads are allowed.
- The path is a regular file.
- The current process can write the file.
- The path is not a mount point.
- The remote source filename is unambiguous.

Recovery downloads a fresh copy of the same repository artifact with forced
download semantics. The fresh bytes are written to a temporary file in the
target directory and atomically installed with `os.replace`, so a failed or
interrupted download never leaves a partial target file. The artifact is then
loaded and smoke-tested once more. If recovery fails, the loader continues to
the next candidate exactly as it does today.

For `HEADROOM_KOMPRESS_ONNX_PATH`, the remote filename comes from
`HEADROOM_KOMPRESS_ONNX_FILENAME` when present. Otherwise, only recognized
default artifact basenames are eligible for replacement. Arbitrary explicit
paths are not overwritten because their corresponding Hub artifact cannot be
identified safely.

Cache-only startup (`allow_download=False`) never attempts recovery. Existing
read-only, mounted, missing, and unsupported-runtime behavior remains
fail-safe.

## Testing

Tests will cover:

- A failed local artifact being refreshed and succeeding on retry.
- A fresh download failure leaving the original artifact unchanged.
- No replacement for mounted or unwritable files.
- No replacement during cache-only loading.
- Existing fallback to the next ONNX candidate after recovery is unavailable
  or unsuccessful.
