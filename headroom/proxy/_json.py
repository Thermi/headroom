"""Fast JSON serialization/deserialization with optional orjson backend.

Provides ``loads``, ``dumps``, and ``JSONDecodeError`` with the same
signatures as the standard library ``json`` module, but backed by
``orjson`` (a Rust-based JSON library that releases the GIL) when
available.
"""

from __future__ import annotations

import json as _stdlib_json
from typing import Any

try:
    import orjson as _orjson

    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

if HAS_ORJSON:
    JSONDecodeError = _orjson.JSONDecodeError

    def loads(data: str | bytes, **kwargs: Any) -> Any:
        if isinstance(data, str):
            data = data.encode("utf-8")
        return _orjson.loads(data)

    def dumps(obj: Any, **kwargs: Any) -> str:
        # orjson uses compact separators (",", ":") by default and emits
        # UTF-8 (like ensure_ascii=False). We only use orjson when the
        # caller's kwargs are compatible; otherwise we fall back to
        # stdlib json.
        option = 0
        indent = kwargs.get("indent")
        if indent is not None:
            option |= _orjson.OPT_INDENT_2
        if kwargs.get("sort_keys", False):
            option |= _orjson.OPT_SORT_KEYS
        default = kwargs.get("default")
        separators = kwargs.get("separators")
        non_compact = separators is not None and separators != (",", ":")
        if not non_compact:
            if default is not None:
                raw: bytes = _orjson.dumps(obj, option=option, default=default)
            else:
                raw = _orjson.dumps(obj, option=option)
            return raw.decode("utf-8")
        _result: str = _stdlib_json.dumps(obj, **kwargs)
        return _result

    def dump(obj: Any, fp: Any, **kwargs: Any) -> None:
        fp.write(dumps(obj, **kwargs))

    def load(fp: Any) -> Any:
        return _orjson.loads(fp.read())


else:
    JSONDecodeError = _stdlib_json.JSONDecodeError

    def loads(data: str | bytes, **kwargs: Any) -> Any:
        return _stdlib_json.loads(data, **kwargs)

    def dumps(obj: Any, **kwargs: Any) -> str:
        _result: str = _stdlib_json.dumps(obj, **kwargs)
        return _result

    def dump(obj: Any, fp: Any, **kwargs: Any) -> None:
        _stdlib_json.dump(obj, fp, **kwargs)

    def load(fp: Any) -> Any:
        return _stdlib_json.load(fp)
