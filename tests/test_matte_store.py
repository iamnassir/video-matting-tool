"""Unit tests for the raw-matte cache (resume / re-render / backward tracking store)."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.errors import StorageError
from app.processing.matte_store import MatteStore, decode_matte, encode_matte, fingerprint_hash

W, H = 320, 180


def make_matte(i: int, dtype=np.uint8) -> np.ndarray:
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((xx - (60 + 2 * i)) ** 2 + (yy - 90) ** 2)
    a = np.clip((50 - r) / 12.0, 0.0, 1.0)
    maxv = np.iinfo(dtype).max
    return (a * maxv + 0.5).astype(dtype)


@pytest.fixture()
def fp():
    return {"input": "clip.mp4", "size": 123, "model": "rvm"}


def test_encode_decode_roundtrip_and_empty():
    m = make_matte(3)
    rec = encode_matte(m)
    assert rec[2] < W and rec[3] < H            # cropped to the subject
    assert np.array_equal(decode_matte(rec, W, H, np.uint8), m)
    empty = encode_matte(np.zeros((H, W), np.uint8))
    assert empty == (0, 0, 0, 0, b"")
    assert not decode_matte(empty, W, H, np.uint8).any()


def test_full_frame_and_16bit_roundtrip():
    rng = np.random.default_rng(0)
    m8 = rng.integers(1, 256, size=(H, W), dtype=np.uint8)       # no zeros: bounding box = full frame
    assert np.array_equal(decode_matte(encode_matte(m8), W, H, np.uint8), m8)
    m16 = rng.integers(0, 65536, size=(H, W), dtype=np.uint16)
    assert np.array_equal(decode_matte(encode_matte(m16), W, H, np.uint16), m16)


def test_decode_rejects_damaged_records():
    x0, y0, w, h, payload = encode_matte(make_matte(0))
    with pytest.raises(ValueError):
        decode_matte((x0, y0, w + 1, h, payload), W, H, np.uint8)       # wrong size
    with pytest.raises(ValueError):
        decode_matte((W - 1, y0, w, h, payload), W, H, np.uint8)        # outside the frame


def test_put_get_flush_reopen(tmp_path, fp):
    store = MatteStore(tmp_path / "m", W, H, fp, chunk_frames=8)
    assert not store.reused
    for i in range(20):
        store.put(i, make_matte(i))
    assert store.contiguous_end(0) == 20 and len(store) == 20
    for i in range(20):                                    # readable before any explicit flush
        assert np.array_equal(store.get(i), make_matte(i))
    store.close()

    again = MatteStore(tmp_path / "m", W, H, fp, chunk_frames=8)
    assert again.reused and len(again) == 20
    assert np.array_equal(again.get(19), make_matte(19))   # the incomplete last chunk was flushed on close
    with pytest.raises(KeyError):
        again.get(20)
    again.close()


def test_descending_writes_and_coverage(tmp_path, fp):
    store = MatteStore(tmp_path / "m", W, H, fp, chunk_frames=8)
    for i in range(29, 19, -1):                            # backward tracking writes in descending order
        store.put(i, make_matte(i))
    assert store.covers(20, 30) and not store.covers(0, 30)
    assert store.contiguous_end(0) == 0 and store.contiguous_end(20) == 30
    store.flush()
    assert store.indices() == list(range(20, 30))
    store.close()


def test_overwrite_and_drop_from(tmp_path, fp):
    store = MatteStore(tmp_path / "m", W, H, fp, chunk_frames=8)
    for i in range(16):
        store.put(i, make_matte(i))
    store.put(3, make_matte(40))                           # overwrite a frame that is already on disk
    store.flush()
    assert np.array_equal(store.get(3), make_matte(40))
    store.drop_from(10)
    assert store.indices() == list(range(10))
    store.close()
    again = MatteStore(tmp_path / "m", W, H, fp, chunk_frames=8)
    assert again.indices() == list(range(10))
    assert np.array_equal(again.get(3), make_matte(40))
    again.close()


def test_fingerprint_change_resets(tmp_path, fp):
    store = MatteStore(tmp_path / "m", W, H, fp)
    store.put(0, make_matte(0))
    store.close()
    other = MatteStore(tmp_path / "m", W, H, {**fp, "model": "sam2"})
    assert not other.reused and len(other) == 0
    assert other.reset_reason and "changed" in other.reset_reason
    other.close()
    assert fingerprint_hash({"a": 1, "b": 2}) == fingerprint_hash({"b": 2, "a": 1})


def test_damaged_chunk_is_discarded(tmp_path, fp):
    d = tmp_path / "m"
    store = MatteStore(d, W, H, fp, chunk_frames=4)
    for i in range(8):
        store.put(i, make_matte(i))
    store.close()
    chunk = sorted(d.glob("chunk_*.mattes"))[1]
    chunk.write_bytes(chunk.read_bytes()[:-7])             # truncate: simulates a crash mid-write
    again = MatteStore(d, W, H, fp, chunk_frames=4)
    assert again.indices() == [0, 1, 2, 3]                 # the damaged chunk is dropped, the rest survives
    again.close()


def test_last_nonempty_before(tmp_path, fp):
    store = MatteStore(tmp_path / "m", W, H, fp, chunk_frames=8)
    for i in range(5):
        store.put(i, make_matte(i))
    for i in range(5, 9):
        store.put(i, np.zeros((H, W), np.uint8))           # subject left the frame
    assert store.last_nonempty_before(9) == 4
    assert store.last_nonempty_before(9, max_lookback=2) is None
    store.close()


def test_rejects_wrong_shape_dtype_and_closed(tmp_path, fp):
    store = MatteStore(tmp_path / "m", W, H, fp)
    with pytest.raises(ValueError):
        store.put(0, np.zeros((H, W + 1), np.uint8))
    with pytest.raises(ValueError):
        store.put(0, np.zeros((H, W), np.uint16))
    store.close()
    with pytest.raises(StorageError):
        store.put(0, make_matte(0))


def test_delete_removes_directory(tmp_path, fp):
    d = tmp_path / "m"
    store = MatteStore(d, W, H, fp, bits=16)
    store.put(2, make_matte(2, np.uint16))
    store.flush()
    assert np.array_equal(store.get(2), make_matte(2, np.uint16))
    store.delete()
    assert not d.exists()
