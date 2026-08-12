"""Image models must be loaded once and reused, not rebuilt per request (#2513).

Every image request built a new ImageCompressor and a new OnnxTechniqueRouter,
each loading native ort.InferenceSession models that grew worker RSS to 1+ GB
over a day. These tests pin the caching / singleton behavior with the heavy
model construction mocked out.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from headroom.image.compressor import ImageCompressor


def test_trained_router_is_built_once_and_cached() -> None:
    compressor = ImageCompressor()
    fake_router = MagicMock(name="TrainedRouter")

    with patch("headroom.image.trained_router.TrainedRouter", return_value=fake_router) as ctor:
        first = compressor._get_router()
        second = compressor._get_router()

    assert first is second is fake_router
    assert ctor.call_count == 1


def test_close_releases_models() -> None:
    compressor = ImageCompressor()
    router = MagicMock()
    compressor._router = router

    compressor.close()

    router.release_models.assert_called_once()
    assert compressor._router is None


def test_get_image_compressor_returns_lightweight_instances() -> None:
    import headroom.proxy.helpers as helpers

    helpers._image_compressor_available = None
    try:
        a = helpers._get_image_compressor()
        b = helpers._get_image_compressor()
        assert a is not None
        assert b is not None
        assert a is not b
    finally:
        helpers._image_compressor_available = None


def test_worker_compressor_is_reused_across_calls() -> None:
    import headroom.proxy.image_isolation as iso

    iso._WORKER_COMPRESSOR = None
    try:
        a = iso._get_worker_compressor()
        b = iso._get_worker_compressor()
        assert a is b
        assert a._is_singleton is True
    finally:
        iso._WORKER_COMPRESSOR = None
