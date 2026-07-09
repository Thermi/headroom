from __future__ import annotations

import os
import ssl
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from headroom.proxy.ssl_context import (
    _additive_ca_context,
    _default_strict_relaxed_context,
    _relax_x509_strict_for_custom_ca,
    _replacement_ca_context,
    apply_global_tls_relaxation,
    build_httpx_verify,
    find_ca_bundle,
    tls_strict_disabled,
)


class TestTlsStrictDisabled:
    def test_default_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert tls_strict_disabled() is False

    def test_disabled_with_zero(self) -> None:
        with patch.dict(os.environ, {"HEADROOM_TLS_STRICT": "0"}, clear=True):
            assert tls_strict_disabled() is True

    def test_disabled_with_false(self) -> None:
        with patch.dict(os.environ, {"HEADROOM_TLS_STRICT": "false"}, clear=True):
            assert tls_strict_disabled() is True

    def test_disabled_with_no(self) -> None:
        with patch.dict(os.environ, {"HEADROOM_TLS_STRICT": "no"}, clear=True):
            assert tls_strict_disabled() is True

    def test_disabled_with_off(self) -> None:
        with patch.dict(os.environ, {"HEADROOM_TLS_STRICT": "off"}, clear=True):
            assert tls_strict_disabled() is True

    def test_enabled_with_one(self) -> None:
        with patch.dict(os.environ, {"HEADROOM_TLS_STRICT": "1"}, clear=True):
            assert tls_strict_disabled() is False

    def test_enabled_with_true(self) -> None:
        with patch.dict(os.environ, {"HEADROOM_TLS_STRICT": "true"}, clear=True):
            assert tls_strict_disabled() is False


class TestRelaxX509Strict:
    def test_relax_when_flag_set(self) -> None:
        ctx = ssl.create_default_context()
        strict_flag = getattr(ssl, "VERIFY_X509_STRICT", 0)
        if not strict_flag:
            pytest.skip("VERIFY_X509_STRICT not available on this platform")
        ctx.verify_flags |= strict_flag
        _relax_x509_strict_for_custom_ca(ctx, path="/fake/ca.pem")
        assert not (ctx.verify_flags & strict_flag)

    def test_noop_when_flag_not_set(self) -> None:
        ctx = ssl.create_default_context()
        strict_flag = getattr(ssl, "VERIFY_X509_STRICT", 0)
        if not strict_flag:
            pytest.skip("VERIFY_X509_STRICT not available on this platform")
        ctx.verify_flags &= ~strict_flag
        result = _relax_x509_strict_for_custom_ca(ctx, path="/fake/ca.pem")
        assert result is ctx


class TestCaContextBuilders:
    def test_replacement_ca_context_sets_alpn(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as f:
            f.write(b"")
            path = f.name
        try:
            with patch("ssl.create_default_context") as mock_create:
                mock_ctx = MagicMock(spec=ssl.SSLContext)
                mock_create.return_value = mock_ctx
                ctx = _replacement_ca_context(path)
                assert ctx is mock_ctx
                mock_ctx.set_alpn_protocols.assert_called_once_with(["h2", "http/1.1"])
        finally:
            os.unlink(path)

    def test_additive_ca_context_loads_verify_locations(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as f:
            f.write(b"")
            path = f.name
        try:
            with patch("ssl.create_default_context") as mock_create:
                mock_ctx = MagicMock(spec=ssl.SSLContext)
                mock_create.return_value = mock_ctx
                ctx = _additive_ca_context(path)
                assert ctx is mock_ctx
                mock_ctx.load_verify_locations.assert_called_once_with(cafile=path)
                mock_ctx.set_alpn_protocols.assert_called_once_with(["h2", "http/1.1"])
        finally:
            os.unlink(path)


class TestFindCaBundle:
    def test_no_env_vars_returns_none(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert find_ca_bundle() is None

    def test_ssl_cert_file_replacement(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as f:
            f.write(b"")
            path = f.name
        try:
            with (
                patch.dict(os.environ, {"SSL_CERT_FILE": path}, clear=True),
                patch("headroom.proxy.ssl_context._replacement_ca_context") as mock_builder,
            ):
                mock_ctx = MagicMock()
                mock_builder.return_value = mock_ctx
                result = find_ca_bundle()
                assert result is mock_ctx
                mock_builder.assert_called_once_with(path)
        finally:
            os.unlink(path)

    def test_requests_ca_bundle_replacement(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as f:
            f.write(b"")
            path = f.name
        try:
            with (
                patch.dict(os.environ, {"REQUESTS_CA_BUNDLE": path}, clear=True),
                patch("headroom.proxy.ssl_context._replacement_ca_context") as mock_builder,
            ):
                result = find_ca_bundle()
                assert result is not None
                mock_builder.assert_called_once_with(path)
        finally:
            os.unlink(path)

    def test_replacement_vars_priority(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as f:
            f.write(b"")
            ssl_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as f:
            f.write(b"")
            req_path = f.name
        try:
            with (
                patch.dict(
                    os.environ,
                    {"SSL_CERT_FILE": ssl_path, "REQUESTS_CA_BUNDLE": req_path},
                    clear=True,
                ),
                patch("headroom.proxy.ssl_context._replacement_ca_context") as mock_builder,
            ):
                find_ca_bundle()
                # SSL_CERT_FILE wins (first in priority)
                mock_builder.assert_called_once_with(ssl_path)
        finally:
            os.unlink(ssl_path)
            os.unlink(req_path)

    def test_node_extra_ca_certs_additive(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as f:
            f.write(b"")
            path = f.name
        try:
            with (
                patch.dict(os.environ, {"NODE_EXTRA_CA_CERTS": path}, clear=True),
                patch("headroom.proxy.ssl_context._additive_ca_context") as mock_builder,
            ):
                result = find_ca_bundle()
                assert result is not None
                mock_builder.assert_called_once_with(path)
        finally:
            os.unlink(path)

    def test_missing_file_skipped(self) -> None:
        with patch.dict(os.environ, {"SSL_CERT_FILE": "/nonexistent/ca.pem"}, clear=True):
            result = find_ca_bundle()
            assert result is None


class TestBuildHttpxVerify:
    def test_default_true(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert build_httpx_verify() is True

    def test_ca_bundle_wins(self) -> None:
        with (
            patch("headroom.proxy.ssl_context.find_ca_bundle") as mock_find,
            patch.dict(os.environ, {}, clear=True),
        ):
            mock_ctx = MagicMock()
            mock_find.return_value = mock_ctx
            result = build_httpx_verify()
            assert result is mock_ctx

    def test_strict_disabled_returns_relaxed_context(self) -> None:
        with (
            patch("headroom.proxy.ssl_context.find_ca_bundle") as mock_find,
            patch("headroom.proxy.ssl_context.tls_strict_disabled") as mock_strict,
            patch("headroom.proxy.ssl_context._default_strict_relaxed_context") as mock_relaxed,
            patch.dict(os.environ, {}, clear=True),
        ):
            mock_find.return_value = None
            mock_strict.return_value = True
            mock_ctx = MagicMock()
            mock_relaxed.return_value = mock_ctx
            result = build_httpx_verify()
            assert result is mock_ctx


class TestApplyGlobalTlsRelaxation:
    def test_noop_when_strict_enabled(self) -> None:
        with patch("headroom.proxy.ssl_context.tls_strict_disabled", return_value=False):
            assert apply_global_tls_relaxation() is False

    def test_noop_when_no_strict_flag(self) -> None:
        with (
            patch("headroom.proxy.ssl_context.tls_strict_disabled", return_value=True),
            patch("ssl.VERIFY_X509_STRICT", 0, create=True),
        ):
            assert apply_global_tls_relaxation() is False

    def test_patches_urllib3_context(self) -> None:
        strict_flag = getattr(ssl, "VERIFY_X509_STRICT", 0)
        if not strict_flag:
            pytest.skip("VERIFY_X509_STRICT not available")

        import urllib3.util.ssl_ as u3ssl

        if getattr(u3ssl.create_urllib3_context, "_headroom_strict_relaxed", False):
            # Already patched by another test sequence — skip to avoid
            # double-patch interactions.
            pytest.skip("urllib3 already patched by prior test")

        with patch("headroom.proxy.ssl_context.tls_strict_disabled", return_value=True):
            result = apply_global_tls_relaxation()
            assert result is True
            assert getattr(u3ssl.create_urllib3_context, "_headroom_strict_relaxed", False)

    def test_idempotent(self) -> None:
        strict_flag = getattr(ssl, "VERIFY_X509_STRICT", 0)
        if not strict_flag:
            pytest.skip("VERIFY_X509_STRICT not available")

        with patch("headroom.proxy.ssl_context.tls_strict_disabled", return_value=True):
            r1 = apply_global_tls_relaxation()
            r2 = apply_global_tls_relaxation()
            assert r1 is True
            assert r2 is True


class TestDefaultStrictRelaxedContext:
    def test_clears_strict_flag(self) -> None:
        strict_flag = getattr(ssl, "VERIFY_X509_STRICT", 0)
        if not strict_flag:
            pytest.skip("VERIFY_X509_STRICT not available on this platform")
        ctx = _default_strict_relaxed_context()
        assert not (ctx.verify_flags & strict_flag)

    def test_sets_alpn(self) -> None:
        ctx = _default_strict_relaxed_context()
        # Just verify it doesn't crash and returns a valid context
        assert isinstance(ctx, ssl.SSLContext)
