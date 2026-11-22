"""REST API tests: the same application object the web interface talks to, driven in-process.

Jobs run through the real pipeline with the weight-free test model on the CPU.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.helpers import media

pytestmark = pytest.mark.usefixtures("test_models")


@pytest.fixture()
def client(app_settings):
    from app.api.server import create_app

    app = create_app(app_settings)
    with TestClient(app) as c:
        yield c


def upload(client, clip) -> dict:
    data = clip.path.read_bytes()
    r = client.post(
        "/api/videos/upload", params={"filename": clip.path.name}, content=data,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert r.status_code == 200, r.text
    return r.json()


def wait_for(client, job_id: str, timeout: float = 180.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} did not finish: {job}")


TEST_CONFIG = {"model": "test_chroma", "device": "cpu"}


# --------------------------------------------------------------------------- static information


def test_health_and_options(client):
    assert client.get("/api/health").json()["status"] == "ok"
    options = client.get("/api/options").json()
    assert options["defaults"]["background"] == "green"
    assert {q["id"] for q in options["quality_presets"]} == {"fast", "balanced", "high", "max"}
    assert {"off", "light", "balanced", "strong"} <= set(options["refinement_presets"])
    codecs = {c["id"]: c for c in options["codecs"]}
    assert codecs["h264"]["available"] and codecs["prores_4444"]["supports_alpha"]
    assert any(b["id"] == "green" and b["color"] == "#00FF00" for b in options["backgrounds"])


def test_system_and_models(client):
    system = client.get("/api/system").json()
    assert system["ffmpeg"]["available"] and system["devices"] and "version" in system
    stats = client.get("/api/system/stats").json()
    assert "ram_total_mb" in stats and stats["busy"] is False
    models = {m["id"]: m for m in client.get("/api/models").json()}
    assert "rvm" in models and "test_chroma" in models
    rvm = models["rvm"]
    assert rvm["license"]["commercial_use"] == "conditional" and rvm["limitations"]
    assert "status" in rvm and "variants" in rvm["status"]


def test_unknown_routes_and_ids_are_json_errors(client):
    r = client.get("/api/does/not/exist")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    r = client.get("/api/jobs/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "job_not_found"
    r = client.get("/api/videos/nope/frame")
    assert r.status_code == 404


def test_model_download_requires_licence_acknowledgement(client):
    r = client.post("/api/models/rvm/download", json={})
    assert r.status_code == 403 and r.json()["error"]["code"] == "license_restriction"


# --------------------------------------------------------------------------- videos


def test_upload_probe_frame_and_delete(client, clip_cfr, app_settings):
    video = upload(client, clip_cfr)
    meta = video["metadata"]
    assert meta["frame_count"] == clip_cfr.frames and meta["video"]["width"] == clip_cfr.width
    assert meta["has_audio"] and meta["is_vfr"] is False
    stored = Path(video["path"])
    assert stored.is_file() and app_settings.data_dir in stored.parents       # stored inside the data directory

    r = client.get(f"/api/videos/{video['id']}/frame", params={"index": 5, "max_side": 160})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.content[:2] == b"\xff\xd8"
    r = client.get(f"/api/videos/{video['id']}/frame", params={"index": 10_000})
    assert r.status_code == 400

    assert [v["id"] for v in client.get("/api/videos").json()] == [video["id"]]
    assert client.delete(f"/api/videos/{video['id']}").status_code == 200
    assert client.get("/api/videos").json() == [] and not stored.exists()


def test_upload_rejects_non_video(client):
    r = client.post("/api/videos/upload", params={"filename": "notes.txt"}, content=b"hello")
    assert r.status_code == 400 and r.json()["error"]["code"] == "unsupported_format"
    r = client.post("/api/videos/upload", params={"filename": "fake.mp4"}, content=b"this is not a video at all")
    assert r.status_code == 400
    assert client.get("/api/videos").json() == []                              # nothing left behind


def test_import_by_path_does_not_copy(client, clip_vfr):
    r = client.post("/api/videos/import", json={"path": str(clip_vfr.path)})
    assert r.status_code == 200, r.text
    video = r.json()
    assert video["imported"] and Path(video["path"]) == clip_vfr.path.resolve()
    assert video["metadata"]["is_vfr"] is True
    client.delete(f"/api/videos/{video['id']}")
    assert clip_vfr.path.is_file()                                             # the user's file is never deleted
    r = client.post("/api/videos/import", json={"path": "Z:/definitely/not/here.mp4"})
    assert r.status_code == 400


def test_proxy_preview_becomes_available(client, clip_rotated):
    video = upload(client, clip_rotated)
    deadline = time.time() + 60
    state = video
    while time.time() < deadline and state["proxy"] == "pending":
        time.sleep(0.2)
        state = client.get(f"/api/videos/{video['id']}").json()
    assert state["proxy"] == "ready", state
    r = client.get(state["preview_url"], headers={"Range": "bytes=0-99"})
    assert r.status_code in (200, 206) and r.headers["content-type"] == "video/mp4"


def test_subject_preview(client, clip_two_subjects):
    video = upload(client, clip_two_subjects)
    w, h = clip_two_subjects.width, clip_two_subjects.height
    box = [w * 0.88 - 22, h * 0.18 - 22, w * 0.88 + 22, h * 0.18 + 22]
    r = client.post(f"/api/videos/{video['id']}/subject-preview",
                    json={"prompt": {"mode": "box", "box": box}, "model": "test_chroma"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["width"] == w and body["height"] == h and body["box"] is not None
    assert body["source"]
    r = client.post(f"/api/videos/{video['id']}/subject-preview", json={"prompt": {"mode": "box"}})
    assert r.status_code == 400                                                # box mode without a box


def test_subject_preview_with_a_pasted_cutout_picture(client, clip_cfr, tmp_path):
    """The path is pasted with quotes and the picture is a colour cut-out of frame 20, not of frame 0."""

    import cv2
    import numpy as np

    video = upload(client, clip_cfr)
    w, h, n, shown = clip_cfr.width, clip_cfr.height, clip_cfr.frames, 20
    frame = media.synth_frame(shown, w, h, n)
    cx, cy = media.subject_center(shown, w, h, n)
    r = media.subject_radius(w, h) * 1.6
    y0, y1, x0, x1 = max(0, int(cy - r)), int(cy + r), max(0, int(cx - r)), int(cx + r)
    cutout = np.empty_like(frame)
    cutout[:] = (20, 40, 235)
    cutout[y0:y1, x0:x1] = frame[y0:y1, x0:x1]
    path = tmp_path / "cut out.png"
    cv2.imencode(".png", cv2.cvtColor(cutout, cv2.COLOR_RGB2BGR))[1].tofile(str(path))
    url = f"/api/videos/{video['id']}/subject-preview"

    res = client.post(url, json={"prompt": {"mode": "mask", "mask_path": f'"{path}"'}, "model": "test_chroma"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["has_mask"] and body["source"] == "user:mask"
    assert body["frame_index"] == shown                                        # moved from frame 0
    assert any(f"shows frame {shown}" in note for note in body["notes"])
    assert 0.02 < body["area_fraction"] < 0.5

    res = client.post(url, json={"prompt": {"mode": "mask", "mask_path": str(path)}, "model": "rvm"})
    assert res.status_code == 200, res.text
    assert any("people only" in note for note in res.json()["notes"])           # a car needs another model

    res = client.post(url, json={"prompt": {"mode": "mask", "mask_path": f'"{tmp_path / "missing.png"}"'}})
    assert res.status_code in (400, 404) and '\\"' not in res.text              # named without the quotes


# --------------------------------------------------------------------------- jobs


def test_job_lifecycle(client, clip_cfr):
    video = upload(client, clip_cfr)
    r = client.post("/api/jobs", json={"video_id": video["id"], "config": {**TEST_CONFIG, "exports": {"alpha_matte": True}}})
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["status"] in ("queued", "running") and job["output_name"].endswith("_green.mp4")

    done = wait_for(client, job["id"])
    assert done["status"] in ("completed", "completed_with_warnings"), done
    assert done["progress"]["stage"] == "done" and done["summary"]["frames"] == clip_cfr.frames
    assert done["summary"]["checks_failed"] == 0
    kinds = {o["kind"] for o in done["outputs"]}
    assert {"main", "alpha_matte"} <= kinds

    main = [o for o in done["outputs"] if o["kind"] == "main"][0]
    r = client.get(main["url"])
    assert r.status_code == 200 and len(r.content) == main["size_bytes"]
    r = client.get(main["url"], headers={"Range": "bytes=0-1023"})
    assert r.status_code == 206 and len(r.content) == 1024

    report = client.get(done["report_url"]).json()
    assert report["timing"]["frames_processed"] == clip_cfr.frames and report["checks"]
    assert "PROCESSING REPORT" in client.get(f"/api/jobs/{job['id']}/report.txt").text
    assert client.get(f"/api/jobs/{job['id']}/log").status_code == 200

    r = client.get(done["preview_url"])
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    assert client.get(f"/api/jobs/{job['id']}/live.jpg").status_code in (200, 204)

    # path traversal is refused
    assert client.get(f"/api/jobs/{job['id']}/files/..%2Fjob.json").status_code == 404

    assert client.delete(f"/api/jobs/{job['id']}").status_code == 200
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404


def test_events_stream_reports_progress(client, clip_cfr):
    video = upload(client, clip_cfr)
    job = client.post("/api/jobs", json={"video_id": video["id"], "config": TEST_CONFIG}).json()
    kinds, frames = [], []
    with client.stream("GET", f"/api/jobs/{job['id']}/events") as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = json.loads(line[5:])
            kinds.append(payload["type"])
            frames.append(payload["job"]["progress"].get("frame", 0))
            if payload["type"] == "finished":
                break
    assert kinds[0] == "snapshot" and kinds[-1] == "finished"
    final = client.get(f"/api/jobs/{job['id']}").json()
    assert final["status"] in ("completed", "completed_with_warnings")


def test_rerender_reuses_the_matte_cache(client, clip_cfr):
    video = upload(client, clip_cfr)
    first = wait_for(client, client.post("/api/jobs", json={"video_id": video["id"], "config": TEST_CONFIG}).json()["id"])
    assert first["summary"]["frames_from_cache"] == 0
    r = client.post(f"/api/jobs/{first['id']}/rerender", json={"config": {"background": "blue", "refinement": {"edge_feather": 1.5}}})
    assert r.status_code == 200, r.text
    second = wait_for(client, r.json()["id"])
    assert second["status"] in ("completed", "completed_with_warnings")
    assert second["summary"]["frames_from_cache"] == clip_cfr.frames           # no inference the second time
    assert second["cache_job_id"] == first["id"] and second["output_name"].endswith("_blue.mp4")


def test_batch_and_queue_order(client, clip_cfr, clip_vfr):
    a, b = upload(client, clip_cfr), upload(client, clip_vfr)
    r = client.post("/api/jobs/batch", json={"video_ids": [a["id"], b["id"]], "config": TEST_CONFIG})
    assert r.status_code == 200 and len(r.json()) == 2
    results = [wait_for(client, j["id"]) for j in r.json()]
    assert all(j["status"] in ("completed", "completed_with_warnings") for j in results)
    assert len(client.get("/api/jobs").json()) == 2
    r = client.post("/api/jobs/batch", json={"video_ids": [a["id"], "missing"], "config": TEST_CONFIG})
    assert r.status_code == 404 and len(client.get("/api/jobs").json()) == 2   # nothing queued when one id is bad


def test_invalid_job_requests(client, clip_cfr):
    video = upload(client, clip_cfr)
    r = client.post("/api/jobs", json={"video_id": video["id"], "config": {"quality": "ultra"}})
    assert r.status_code == 400 and "quality" in r.json()["error"]["message"]
    r = client.post("/api/jobs", json={"video_id": video["id"], "config": {"background": "transparent", "encoding": {"codec": "h264"}}})
    assert r.status_code == 400 and "alpha" in r.json()["error"]["message"]
    r = client.post("/api/jobs", json={"video_id": "missing", "config": {}})
    assert r.status_code == 404


def test_failed_job_reports_the_error(client, clip_cfr):
    video = upload(client, clip_cfr)
    job = client.post("/api/jobs", json={"video_id": video["id"], "config": {"model": "does_not_exist", "device": "cpu"}}).json()
    done = wait_for(client, job["id"])
    assert done["status"] == "failed" and done["error"]["code"] == "unknown_model"
    assert done["outputs"] == []
    r = client.post(f"/api/jobs/{job['id']}/resume")
    assert r.status_code == 200
    again = wait_for(client, job["id"])
    assert again["status"] == "failed"


def test_cancel_running_job(client, clip_cfr, test_models):
    import torch

    original = test_models["chroma"].process

    def slow(self, frames, frame_indices):
        time.sleep(0.15)
        return original(self, frames, frame_indices)

    test_models["chroma"].process = slow
    try:
        video = upload(client, clip_cfr)
        job = client.post("/api/jobs", json={"video_id": video["id"], "config": {**TEST_CONFIG, "batch_size": 1}}).json()
        deadline = time.time() + 60
        while time.time() < deadline:
            state = client.get(f"/api/jobs/{job['id']}").json()
            if state["status"] == "running" and state["progress"].get("stage") == "render" and state["progress"].get("frame", 0) >= 3:
                break
            time.sleep(0.05)
        assert client.post(f"/api/jobs/{job['id']}/cancel").status_code == 200
        done = wait_for(client, job["id"])
    finally:
        test_models["chroma"].process = original
    assert done["status"] == "cancelled" and done["outputs"] == []
    resumed = client.post(f"/api/jobs/{job['id']}/resume").json()
    final = wait_for(client, resumed["id"])
    assert final["status"] in ("completed", "completed_with_warnings")
    assert final["summary"]["frames_from_cache"] >= 1                          # the cancelled run's mattes were reused


def test_jobs_survive_a_restart(app_settings, clip_cfr):
    from app.api.server import create_app

    with TestClient(create_app(app_settings)) as c:
        video = upload(c, clip_cfr)
        job = wait_for(c, c.post("/api/jobs", json={"video_id": video["id"], "config": TEST_CONFIG}).json()["id"])
    with TestClient(create_app(app_settings)) as c:                            # a new server on the same data directory
        jobs = c.get("/api/jobs").json()
        assert [j["id"] for j in jobs] == [job["id"]] and jobs[0]["status"] == job["status"]
        assert c.get(f"/api/videos/{video['id']}").status_code == 200
        main = [o for o in jobs[0]["outputs"] if o["kind"] == "main"][0]
        assert c.get(main["url"]).status_code == 200
