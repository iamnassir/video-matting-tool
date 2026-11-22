"""Browser smoke tests of the web interface (Playwright driving headless Microsoft Edge).

The real application (``create_app``) runs with uvicorn on a free loopback port in a background
thread, with its data / cache / temp folders under the test's tmp_path. Nothing is mocked: the
pages talk to the real API.

Skipped cleanly when Playwright or Microsoft Edge is not available. Run with the machine-wide lock
(the job test imports PyTorch):

    python scripts/with_lock.py torch -- python -m pytest tests/test_ui_smoke.py -q
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Iterator, List

import pytest

pytestmark = pytest.mark.ui

VIEWS = ["source", "subject", "settings", "process", "results", "models"]


# --------------------------------------------------------------------------- server


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture()
def ui_server(app_settings) -> Iterator[str]:
    uvicorn = pytest.importorskip("uvicorn")
    from app.api.server import create_app

    app = create_app(app_settings)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on"))
    thread = threading.Thread(target=server.run, name="ui-test-server", daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            pytest.fail("the application server did not start")
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=30)


# --------------------------------------------------------------------------- browser


@pytest.fixture(scope="module")
def browser():
    sync_api = pytest.importorskip("playwright.sync_api", reason="playwright is not installed")
    try:
        manager = sync_api.sync_playwright().start()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"playwright cannot start: {exc}")
    try:
        browser = manager.chromium.launch(channel="msedge", headless=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        manager.stop()
        pytest.skip(f"Microsoft Edge is not available for Playwright: {str(exc).splitlines()[0]}")
    yield browser
    browser.close()
    manager.stop()


class Problems:
    """Console errors, page errors and failed requests seen by a page."""

    def __init__(self) -> None:
        self.items: List[str] = []

    def assert_none(self) -> None:
        assert not self.items, "the page reported problems:\n" + "\n".join(self.items)


@pytest.fixture()
def page(browser, ui_server):
    context = browser.new_context(viewport={"width": 1440, "height": 900})
    page = context.new_page()
    problems = Problems()
    page.on("console", lambda m: problems.items.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
    page.on("pageerror", lambda e: problems.items.append(f"page error: {e}"))
    page.on("response", lambda r: problems.items.append(f"HTTP {r.status} {r.request.method} {r.url}") if r.status >= 400 else None)
    page.on("requestfailed", lambda r: problems.items.append(f"request failed: {r.method} {r.url} {r.failure}"))
    page.problems = problems
    page.base = ui_server
    page.goto(ui_server + "/#/source", wait_until="load")
    page.wait_for_selector("[data-testid=app]")
    yield page
    context.close()


def _upload(page, path) -> None:
    page.goto(page.base + "/#/source")
    # the options (allowed extensions) and the library must be loaded before a file is queued
    page.wait_for_selector("[data-testid=library-empty], [data-testid=video-card]", timeout=20000)
    page.wait_for_function("document.querySelector('[data-testid=app-version]') && !document.querySelector('[data-testid=app-version]').hidden")
    before = page.locator("[data-testid=video-card]").count()
    page.set_input_files("[data-testid=file-input]", str(path))
    page.wait_for_function(f"document.querySelectorAll('[data-testid=video-card]').length > {before}", timeout=60000)


# --------------------------------------------------------------------------- tests


def test_every_view_renders_without_errors(page, app_settings):
    from app.models import registry

    page.wait_for_selector("[data-testid=main-nav]")
    page.wait_for_selector("[data-testid=dropzone]")
    assert page.locator("[data-testid=file-input]").count() == 1
    for view in VIEWS:
        assert page.locator(f"[data-testid=nav-{view}]").count() == 1, view
        page.click(f"[data-testid=nav-{view}]")
        page.wait_for_selector(f"[data-testid=view-{view}]", timeout=10000)
        assert page.evaluate("location.hash") == f"#/{view}"
    # Models lists every registered model (matting and helper models)
    page.click("[data-testid=nav-models]")
    page.wait_for_selector("[data-testid=model-card]", timeout=20000)
    shown = set(page.eval_on_selector_all("[data-testid=model-card]", "els => els.map(e => e.dataset.modelId)"))
    registered = {info.id for info in registry.list_model_infos(app_settings, include_auxiliary=True)}
    assert registered, "no model is registered"
    assert registered <= shown, f"not listed: {sorted(registered - shown)}"
    page.wait_for_timeout(500)
    page.problems.assert_none()


def test_upload_creates_library_card(page, clip_cfr):
    _upload(page, clip_cfr.path)
    card = page.locator("[data-testid=video-card]").first
    text = card.inner_text()
    assert f"{clip_cfr.width}×{clip_cfr.height}" in text, text
    assert clip_cfr.path.name in text or "cfr" in text
    page.problems.assert_none()


def test_pasted_cutout_picture_moves_the_reference_frame(test_models, clip_cfr, page, tmp_path):
    """Mask image: a path pasted with quotes, a colour cut-out of frame 20 while frame 0 is selected."""

    import cv2
    import numpy as np

    from tests.helpers import media

    w, h, n, shown = clip_cfr.width, clip_cfr.height, clip_cfr.frames, 20
    frame = media.synth_frame(shown, w, h, n)
    cx, cy = media.subject_center(shown, w, h, n)
    r = media.subject_radius(w, h) * 1.6
    cutout = np.empty_like(frame)
    cutout[:] = (20, 40, 235)
    y0, y1, x0, x1 = max(0, int(cy - r)), int(cy + r), max(0, int(cx - r)), int(cx + r)
    cutout[y0:y1, x0:x1] = frame[y0:y1, x0:x1]
    picture = tmp_path / "wanted look.png"
    cv2.imencode(".png", cv2.cvtColor(cutout, cv2.COLOR_RGB2BGR))[1].tofile(str(picture))

    _upload(page, clip_cfr.path)
    page.click("[data-testid=nav-subject]")
    page.wait_for_selector("[data-testid=subject-tool-mask]", timeout=20000)
    page.click("[data-testid=subject-tool-mask]")
    page.fill("[data-testid=subject-mask-path]", f'"{picture}"')          # as Explorer's "Copy as path" gives it
    assert page.inner_text("[data-testid=subject-frame-label]").startswith("Frame 0 ")
    page.click("[data-testid=subject-preview]")
    page.wait_for_selector("[data-testid=subject-preview-facts], [data-testid=subject-preview-error]", timeout=60000)
    assert page.locator("[data-testid=subject-preview-error]").count() == 0, page.inner_text("[data-testid=subject-result]")
    facts = page.inner_text("[data-testid=subject-preview-facts]")
    assert f"shows frame {shown}" in facts, facts
    # the page follows: slider, label and the stored selection are on the frame the picture shows
    page.wait_for_function(
        f"document.querySelector('[data-testid=subject-frame-label]').textContent.startsWith('Frame {shown} ')", timeout=10000)
    assert page.input_value("[data-testid=subject-frame-slider]") == str(shown)
    assert page.locator("[data-testid=subject-preview-facts]").count() == 1   # the preview stays valid after the move
    page.wait_for_timeout(600)
    page.problems.assert_none()


def test_job_runs_to_results(test_models, clip_cfr, page):
    # test_models comes first: the adapter must be registered before the page loads /api/models
    try:
        import app.processing.pipeline  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"the engine cannot run jobs yet: {exc}")
    model_id = test_models["chroma"].info.id   # "test_chroma"

    _upload(page, clip_cfr.path)               # the new video is selected automatically
    page.click("[data-testid=nav-settings]")
    option = page.locator(f"[data-testid=model-option-{model_id}]")
    option.wait_for(timeout=20000)
    option.click()
    page.wait_for_function(f"document.querySelector('[data-testid=model-option-{model_id}]').getAttribute('aria-checked') === 'true'")

    page.click("[data-testid=nav-process]")
    start = page.locator("[data-testid=start-job]")
    start.wait_for()
    page.wait_for_function("!document.querySelector('[data-testid=start-job]').disabled", timeout=10000)
    start.click()
    page.wait_for_selector("[data-testid=job-finished], [data-testid=job-error]", timeout=240000)
    assert page.locator("[data-testid=job-error]").count() == 0, page.inner_text("[data-testid=job-monitor]")

    page.click("[data-testid=job-results]")
    page.wait_for_selector("[data-testid=report]", timeout=30000)
    assert page.locator("[data-testid=report-check]").count() > 0
    assert page.locator("[data-testid=report-comparison] tbody tr").count() > 0
    assert page.locator("[data-testid=download-main]").count() == 1
    page.wait_for_function("(() => { const v = document.querySelector('[data-testid=result-player-video]'); return v && v.readyState >= 2 && v.videoWidth > 0; })()", timeout=30000)
    canvas = page.locator("[data-testid=result-player-canvas]")
    assert canvas.is_visible()
    page.problems.assert_none()
