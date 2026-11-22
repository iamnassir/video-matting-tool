"""Model registry, licence policy, plugins and the explicit, verified weight downloads.

Downloads are exercised against a tiny HTTP server on the loopback interface, so the tests need no
internet access and prove the real code path: range resume, size and SHA-256 verification, atomic
placement, cancellation.
"""

from __future__ import annotations

import hashlib
import http.server
import socketserver
import sys
import threading
from pathlib import Path

import pytest

from app.core.config import AppSettings, QualityPreset
from app.core.errors import (
    ChecksumMismatchError,
    ConfigError,
    JobCancelledError,
    LicenseError,
    ModelNotInstalledError,
    UnknownModelError,
    WeightsDownloadError,
    WeightsMissingError,
)
from app.core.events import CancelToken
from app.models import registry
from app.models.base import (
    LicenseInfo,
    MatteOutput,
    MattingModel,
    ModelCapabilities,
    ModelInfo,
    ModelVariant,
    WeightFile,
)
from app.models.weights import WeightsManager, sha256_file

PAYLOAD = bytes(range(256)) * 1024             # 256 KiB of deterministic data
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


# --------------------------------------------------------------------------- a local file server


class _Handler(http.server.BaseHTTPRequestHandler):
    files = {"/weights.bin": PAYLOAD, "/other.bin": b"x" * 1000}
    requests = []
    ignore_range = False

    def do_GET(self):  # noqa: N802 - http.server API
        type(self).requests.append((self.path, self.headers.get("Range")))
        data = self.files.get(self.path)
        if data is None:
            self.send_error(404)
            return
        start = 0
        rng = self.headers.get("Range")
        if rng and not self.ignore_range:
            start = int(rng.split("=")[1].split("-")[0])
            if start >= len(data):
                self.send_error(416)
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            self.send_response(200)
        body = data[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep the test output clean
        pass


@pytest.fixture()
def server():
    _Handler.requests = []
    _Handler.ignore_range = False
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def weight(base, **overrides):
    values = dict(filename="weights.bin", url=f"{base}/weights.bin", size_bytes=len(PAYLOAD), sha256=DIGEST)
    values.update(overrides)
    return WeightFile(**values)


def info_for(weights, model_id="toy", commercial="allowed", requires=()):
    return ModelInfo(
        id=model_id, name="Toy", description="test model",
        license=LicenseInfo(name="Test licence", url="https://example.invalid/licence", commercial_use=commercial,
                            notes="for tests"),
        capabilities=ModelCapabilities(alpha_type="alpha"),
        variants=(ModelVariant(id="default", label="default", weights=tuple(weights)),),
        default_variant="default", requires=tuple(requires), install_hint="pip install the-missing-thing",
    )


# --------------------------------------------------------------------------- downloads


def test_weights_are_never_downloaded_implicitly(tmp_path, server):
    info = info_for([weight(server)])
    manager = WeightsManager(tmp_path)
    with pytest.raises(WeightsMissingError) as raised:
        manager.ensure(info, info.variant(None), download=False)
    assert "models download toy" in raised.value.hint and _Handler.requests == []      # no network access at all
    assert not manager.is_ready("toy", info.variant(None))


def test_download_verifies_and_places_the_file(tmp_path, server):
    info = info_for([weight(server)])
    manager = WeightsManager(tmp_path)
    seen = []
    paths = manager.ensure(info, info.variant(None), download=True, progress=lambda done, total, name: seen.append((done, total)))
    assert paths == [tmp_path / "toy" / "weights.bin"] and paths[0].read_bytes() == PAYLOAD
    assert seen[-1] == (len(PAYLOAD), len(PAYLOAD)) and not (tmp_path / "toy" / "weights.bin.part").exists()
    assert manager.is_ready("toy", info.variant(None))
    assert manager.check_file("toy", info.variant(None).weights[0], full=True)["detail"] == "verified"
    before = len(_Handler.requests)
    manager.ensure(info, info.variant(None), download=True)                             # present: no second request
    assert len(_Handler.requests) == before


def test_interrupted_download_resumes_with_a_range_request(tmp_path, server):
    manager = WeightsManager(tmp_path)
    (tmp_path / "toy").mkdir()
    (tmp_path / "toy" / "weights.bin.part").write_bytes(PAYLOAD[:100_000])
    path = manager.download("toy", weight(server))
    assert path.read_bytes() == PAYLOAD
    assert ("/weights.bin", "bytes=100000-") in _Handler.requests


def test_server_without_range_support_restarts_cleanly(tmp_path, server):
    _Handler.ignore_range = True
    manager = WeightsManager(tmp_path)
    (tmp_path / "toy").mkdir()
    (tmp_path / "toy" / "weights.bin.part").write_bytes(b"garbage" * 1000)
    assert manager.download("toy", weight(server)).read_bytes() == PAYLOAD


def test_checksum_and_size_mismatch_are_rejected(tmp_path, server):
    manager = WeightsManager(tmp_path)
    with pytest.raises(ChecksumMismatchError):
        manager.download("toy", weight(server, sha256="0" * 64))
    assert not (tmp_path / "toy" / "weights.bin").exists() and not (tmp_path / "toy" / "weights.bin.part").exists()
    with pytest.raises(ChecksumMismatchError):
        manager.download("toy", weight(server, filename="other.bin", url=f"{server}/other.bin", size_bytes=5, sha256=None))


def test_missing_file_on_the_server(tmp_path, server):
    manager = WeightsManager(tmp_path)
    with pytest.raises(WeightsDownloadError) as raised:
        manager.download("toy", weight(server, url=f"{server}/nope.bin"), retries=2)
    assert "nope.bin" in raised.value.message
    assert len([r for r in _Handler.requests if r[0] == "/nope.bin"]) == 1              # a 404 is not retried


def test_corrupted_local_file_is_detected_and_replaced(tmp_path, server):
    info = info_for([weight(server)])
    manager = WeightsManager(tmp_path)
    manager.ensure(info, info.variant(None), download=True)
    target = tmp_path / "toy" / "weights.bin"
    damaged = bytearray(PAYLOAD)
    damaged[1234] ^= 0xFF
    target.write_bytes(bytes(damaged))                                                  # same size, different content
    status = manager.check_file("toy", info.variant(None).weights[0], full=True)
    assert not status["ok"] and "SHA-256 mismatch" in status["detail"]
    manager.ensure(info, info.variant(None), download=True)
    assert sha256_file(target) == DIGEST


def test_failed_check_ends_the_trust_in_an_earlier_verification(tmp_path, server):
    """Same size and same modification time, different content: what a quick check cannot see.

    (On file systems with a coarse clock a file rewritten at once keeps its modification time; here
    the time is put back explicitly so that the case is the same on every system.)
    """

    import os

    info = info_for([weight(server)])
    spec = info.variant(None).weights[0]
    manager = WeightsManager(tmp_path)
    manager.ensure(info, info.variant(None), download=True)
    target = tmp_path / "toy" / "weights.bin"
    marker = target.with_name(target.name + ".verified.json")
    assert marker.is_file()
    before = target.stat()
    damaged = bytearray(PAYLOAD)
    damaged[4321] ^= 0x01
    target.write_bytes(bytes(damaged))
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert manager.check_file("toy", spec)["ok"]                    # the quick check still trusts the marker
    status = manager.check_file("toy", spec, full=True)             # e.g. "models verify"
    assert not status["ok"] and "SHA-256 mismatch" in status["detail"]
    assert not marker.exists()                                      # ... and now nothing does any more
    assert not manager.check_file("toy", spec)["ok"]
    manager.ensure(info, info.variant(None), download=True)
    assert sha256_file(target) == DIGEST


def test_download_can_be_cancelled(tmp_path, server):
    token = CancelToken()
    token.cancel()
    with pytest.raises(JobCancelledError):
        WeightsManager(tmp_path).download("toy", weight(server), cancel=token)


def test_mirror_override_and_removal(tmp_path, server):
    official = weight("http://127.0.0.1:9")                                            # unreachable "official" URL
    manager = WeightsManager(tmp_path, overrides={"weights.bin": {"url": f"{server}/weights.bin", "sha256": DIGEST,
                                                                  "size_bytes": str(len(PAYLOAD))}})
    assert manager.resolved(official).url.startswith(server)
    info = info_for([official])
    manager.ensure(info, info.variant(None), download=True)
    assert manager.remove("toy", info.variant(None)) >= len(PAYLOAD)
    assert not (tmp_path / "toy" / "weights.bin").exists()


# --------------------------------------------------------------------------- registry and licence policy


class _ToyModel(MattingModel):
    info = info_for([], model_id="toy_registered")

    def load(self):
        self.loaded = True

    def start_sequence(self, seq):
        pass

    def process(self, frames, frame_indices):
        return MatteOutput(alpha=frames[:, :1])


@pytest.fixture()
def clean_registry():
    registry.ensure_loaded()
    before = dict(registry._REGISTRY), dict(registry._AUXILIARY)
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(before[0])
    registry._AUXILIARY.clear()
    registry._AUXILIARY.update(before[1])


def settings_for(tmp_path, **overrides):
    return AppSettings(models_dir=tmp_path / "models", data_dir=tmp_path / "data", cache_dir=tmp_path / "cache", **overrides)


def test_builtin_models_are_registered_with_complete_metadata():
    infos = {i.id: i for i in registry.list_model_infos(include_auxiliary=True)}
    assert "rvm" in infos
    for info in infos.values():
        assert info.license.name and info.license.commercial_use in ("allowed", "conditional", "prohibited")
        assert info.default_variant in [v.id for v in info.variants]
        for variant in info.variants:
            for w in variant.weights:
                assert w.url.startswith("https://"), (info.id, w.url)                   # official sources only
                assert w.sha256 and len(w.sha256) == 64 and w.size_bytes, (info.id, w.filename)
        if info.kind == "matting":
            assert info.limitations, f"{info.id} must state its limitations"
    assert registry.load_errors() == {}


def test_register_model_validation(clean_registry):
    registry.register_model(_ToyModel)
    assert registry.get_model_class("toy_registered") is _ToyModel
    with pytest.raises(TypeError):
        registry.register_model(object)                                               # not an adapter
    with pytest.raises(UnknownModelError) as raised:
        registry.get_model_class("nope")
    assert "rvm" in raised.value.hint


def test_licence_policy(clean_registry, tmp_path):
    class NonCommercial(_ToyModel):
        info = info_for([], model_id="toy_nc", commercial="prohibited")

    class Conditional(_ToyModel):
        info = info_for([], model_id="toy_cond", commercial="conditional")

    registry.register_model(NonCommercial)
    registry.register_model(Conditional)
    settings = settings_for(tmp_path)
    import torch

    cpu = torch.device("cpu")
    notes = []
    model = registry.create_model("toy_nc", settings=settings, device=cpu, warnings=notes)
    assert isinstance(model, NonCommercial) and any("non-commercial" in n for n in notes)
    with pytest.raises(LicenseError) as raised:
        registry.create_model("toy_nc", settings=settings, device=cpu, commercial_use=True)
    assert "commercial" in raised.value.message
    with pytest.raises(LicenseError):                                                  # installation-wide switch
        registry.create_model("toy_nc", settings=settings_for(tmp_path, commercial_use=True), device=cpu)
    notes = []
    registry.create_model("toy_cond", settings=settings, device=cpu, commercial_use=True, warnings=notes)
    assert notes and "Test licence" in notes[0]                                        # allowed, with a notice


def test_missing_python_package_is_reported_with_the_install_hint(clean_registry, tmp_path):
    class NeedsPackage(_ToyModel):
        info = info_for([], model_id="toy_pkg", requires=("a_package_that_does_not_exist_xyz",))

    registry.register_model(NeedsPackage)
    import torch

    with pytest.raises(ModelNotInstalledError) as raised:
        registry.create_model("toy_pkg", settings=settings_for(tmp_path), device=torch.device("cpu"))
    assert "a_package_that_does_not_exist_xyz" in raised.value.message and "pip install" in raised.value.hint
    status = registry.model_status("toy_pkg", settings_for(tmp_path))
    assert not status["installed"] and not status["ready"] and status["install_hint"]


def test_create_model_resolves_variant_precision_and_options(clean_registry, tmp_path):
    registry.register_model(_ToyModel)
    import torch

    model = registry.create_model(
        "toy_registered", settings=settings_for(tmp_path), device=torch.device("cpu"), precision="fp16",
        quality=QualityPreset.HIGH, options={"knob": 3},
    )
    assert model.precision == "fp32"                          # half precision is never used on the CPU
    assert model.options["knob"] == 3 and model.options["quality"] == "high"
    assert model.variant.id == "default" and callable(model.factory)
    with pytest.raises(ConfigError):
        registry.create_model("toy_registered", settings=settings_for(tmp_path), device=torch.device("cpu"), variant="xl")


def test_auxiliary_models_share_weight_management(clean_registry, tmp_path, server):
    helper = info_for([weight(server)], model_id="toy_helper")
    with pytest.raises(ValueError):
        registry.register_auxiliary(helper)                                            # kind must be "auxiliary"
    import dataclasses

    helper = dataclasses.replace(helper, kind="auxiliary")
    registry.register_auxiliary(helper)
    settings = settings_for(tmp_path)
    assert "toy_helper" in [i.id for i in registry.list_model_infos(include_auxiliary=True)]
    assert "toy_helper" not in [i.id for i in registry.list_model_infos()]             # not selectable as --model
    with pytest.raises(UnknownModelError) as raised:
        registry.get_model_class("toy_helper")
    assert "helper model" in raised.value.hint
    with pytest.raises(WeightsMissingError):
        registry.ensure_auxiliary("toy_helper", settings=settings)
    paths = registry.ensure_auxiliary("toy_helper", settings=settings, download=True)
    assert paths["weights.bin"].read_bytes() == PAYLOAD
    assert registry.model_status("toy_helper", settings)["ready"]


def test_plugins_are_loaded_from_the_settings(clean_registry, tmp_path, monkeypatch):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "vmt_test_plugin.py").write_text(
        "from tests.test_registry_weights import _ToyModel, info_for\n"
        "from app.models.registry import register_model\n"
        "class PluginModel(_ToyModel):\n"
        "    info = info_for([], model_id='plugin_model')\n"
        "register_model(PluginModel)\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(plugin_dir))
    registry._plugins_loaded.discard("vmt_test_plugin")
    settings = settings_for(tmp_path, plugins=["vmt_test_plugin", "a_plugin_that_does_not_exist"])
    assert "plugin_model" in registry.available_model_ids(settings)
    assert "rvm" in registry.available_model_ids(settings)                             # a broken plugin harms nothing
    sys.modules.pop("vmt_test_plugin", None)
