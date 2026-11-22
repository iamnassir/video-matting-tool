"""Helper models for subject selection (used by :mod:`app.tracking.subject`).

Three optional backends turn what the user says about the subject into something a matting model
can start from:

=========  =================================================  =====================================
backend    what it does                                       model
=========  =================================================  =====================================
detector   finds the primary subject automatically, or the    torchvision Mask R-CNN (COCO)
           detected instance that matches a box / clicks
sam2       turns a box and / or clicks into a precise mask    SAM 2.1 image predictor (the weights
           on one frame                                       of the ``sam2`` matting model)
text       turns a description ("the woman in red") into a    Grounding DINO tiny (transformers)
           box
=========  =================================================  =====================================

The detector and the text model are registered as *auxiliary* models: their weights are downloaded
explicitly and verified like every other model (``python main.py models download detector``), and
nothing is ever fetched implicitly - torchvision and transformers are only given local files.

Each backend is created on demand and released by ``close()`` before the matting model is loaded,
so the helper never competes with it for GPU memory.  The detector and the text model run on the
CPU by default: they are needed once per clip and CPU time is cheaper than scarce VRAM.
"""

from __future__ import annotations

import contextlib
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..core.config import AppSettings
from ..core.errors import ModelLoadError, ModelNotInstalledError, UnknownModelError, VmtError
from ..core.events import CancelToken
from ..core.types import ResolvedSubject, box_from_mask, clamp_box
from ..models import registry
from ..models.base import LicenseInfo, ModelCapabilities, ModelInfo, ModelVariant, WeightFile

log = logging.getLogger(__name__)

__all__ = ["available", "create", "DETECTOR_ID", "TEXT_ID"]

DETECTOR_ID = "detector"
TEXT_ID = "text_grounding"

Box = Tuple[float, float, float, float]
Point = Tuple[float, float, int]

# --------------------------------------------------------------------------- auxiliary model metadata

_GDINO_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
_GDINO_URL = f"https://huggingface.co/IDEA-Research/grounding-dino-tiny/resolve/{_GDINO_REVISION}"
_GDINO_FILES = (
    ("config.json", 1_644, "eec82c5ab66e16df12a9a212e68ac011779927c2536cf9078658e35d85f0c67a"),
    ("preprocessor_config.json", 457, "8454179ba95e2ad22947835aad7b45862a601fc0055ab88bf1ee70892d3aea60"),
    ("special_tokens_map.json", 125, "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3"),
    ("tokenizer.json", 711_396, "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"),
    ("tokenizer_config.json", 1_237, "d40ab645b68211910b9170d22433d43186a6ec8ee6fd10ba170524b25bf4fb56"),
    ("vocab.txt", 231_508, "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"),
    ("model.safetensors", 689_359_096, "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"),
)

DETECTOR_INFO = registry.register_auxiliary(
    ModelInfo(
        id=DETECTOR_ID,
        kind="auxiliary",
        name="Subject detector (Mask R-CNN)",
        description="Finds the primary subject on the reference frame for automatic selection (people first, then "
        "animals, vehicles and other COCO objects). Runs once per clip, on the CPU.",
        license=LicenseInfo(
            name="BSD-3-Clause (torchvision code); COCO-trained weights",
            spdx="BSD-3-Clause",
            url="https://github.com/pytorch/vision/blob/main/LICENSE",
            commercial_use="conditional",
            notes="The torchvision code is BSD-3-Clause. The pretrained weights carry no explicit licence statement "
            "and were trained on COCO, whose images have their own terms; review this before commercial deployment.",
        ),
        capabilities=ModelCapabilities(alpha_type="mask", subject_domain="any"),
        variants=(
            ModelVariant(
                id="maskrcnn_v2", label="Mask R-CNN ResNet-50 FPN v2 (COCO)",
                weights=(WeightFile(
                    filename="maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth",
                    url="https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth",
                    size_bytes=185_828_065,
                    sha256="73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e",
                    description="torchvision COCO_V1 weights", source="download.pytorch.org (official torchvision weights)",
                ),),
            ),
        ),
        default_variant="maskrcnn_v2",
        limitations=(
            "Knows the 80 COCO object classes only; its masks are coarse and are refined by SAM 2 when available.",
            "With several similar subjects the automatic choice can be ambiguous: select the subject manually then.",
        ),
        requires=("torchvision",),
        install_hint="torchvision is installed together with PyTorch (see docs/INSTALL.md).",
        homepage="https://pytorch.org/vision/stable/models/mask_rcnn.html",
    )
)

TEXT_INFO = registry.register_auxiliary(
    ModelInfo(
        id=TEXT_ID,
        kind="auxiliary",
        name="Text grounding (Grounding DINO tiny)",
        description="Finds the subject from a text description such as 'the woman in the red jacket'. "
        "Runs once per clip, on the CPU.",
        license=LicenseInfo(
            name="Apache-2.0", spdx="Apache-2.0", url="https://huggingface.co/IDEA-Research/grounding-dino-tiny",
            commercial_use="allowed", notes="Model and code are released under Apache-2.0.",
        ),
        capabilities=ModelCapabilities(alpha_type="mask", subject_domain="any"),
        variants=(
            ModelVariant(
                id="tiny", label="Grounding DINO tiny",
                weights=tuple(
                    WeightFile(filename=name, url=f"{_GDINO_URL}/{name}", size_bytes=size, sha256=digest,
                               source=f"Hugging Face IDEA-Research/grounding-dino-tiny @ {_GDINO_REVISION[:8]}")
                    for name, size, digest in _GDINO_FILES
                ),
            ),
        ),
        default_variant="tiny",
        limitations=(
            "Returns a box for the best match of the description: an ambiguous or wrong description still selects "
            "something. Check the selection preview before processing.",
            "Takes several seconds on the CPU (once per clip).",
        ),
        requires=("transformers",),
        install_hint="Install the text-prompt package with:  python scripts/install_models.py text",
        homepage="https://huggingface.co/IDEA-Research/grounding-dino-tiny",
    )
)


# --------------------------------------------------------------------------- availability


def _sam2_variant(settings: AppSettings) -> Optional[str]:
    """The SAM 2 variant to use for single images: the recommended one if present, else any."""

    info = registry.get_model_info("sam2", settings)
    manager = registry.weights_manager(settings)
    for candidate in ("small", "tiny", "base_plus", "large"):
        if manager.is_ready(info.id, info.variant(candidate)):
            return candidate
    return None


def available(settings: AppSettings) -> Dict[str, bool]:
    """Which backends can be used right now: package importable AND weights on disk."""

    out = {"sam2": False, "detector": False, "text": False}
    manager = registry.weights_manager(settings)
    try:
        info = registry.get_model_info("sam2", settings)
        out["sam2"] = not registry.missing_modules(info) and _sam2_variant(settings) is not None
    except VmtError:
        pass
    out["detector"] = not registry.missing_modules(DETECTOR_INFO) and manager.is_ready(DETECTOR_ID, DETECTOR_INFO.variant(None))
    out["text"] = not registry.missing_modules(TEXT_INFO) and manager.is_ready(TEXT_ID, TEXT_INFO.variant(None))
    return out


def create(
    name: str,
    settings: AppSettings,
    device: Any,
    *,
    commercial_use: Optional[bool] = None,
    download: Optional[bool] = None,
    cancel: Optional[CancelToken] = None,
    warnings: Optional[List[str]] = None,
) -> Any:
    """Build a backend, or raise a :class:`VmtError` that says what is missing and how to get it."""

    if name == "detector":
        paths = registry.ensure_auxiliary(DETECTOR_ID, settings=settings, commercial_use=commercial_use, download=download,
                                          cancel=cancel, warnings=warnings)
        return DetectorBackend(next(iter(paths.values())))
    if name == "text":
        paths = registry.ensure_auxiliary(TEXT_ID, settings=settings, commercial_use=commercial_use, download=download,
                                          cancel=cancel, warnings=warnings)
        return TextGroundingBackend(next(iter(paths.values())).parent)
    if name == "sam2":
        info = registry.get_model_info("sam2", settings)
        note = registry.check_license(info, settings.commercial_use if commercial_use is None else bool(commercial_use))
        if note and warnings is not None and note not in warnings:
            warnings.append(note)
        missing = registry.missing_modules(info)
        if missing:
            raise ModelNotInstalledError(
                f"Turning clicks or a box into a mask needs SAM 2, whose package is not installed ({', '.join(missing)}).",
                hint=info.install_hint,
            )
        variant = _sam2_variant(settings)
        manager = registry.weights_manager(settings)
        if variant is None:
            variant = info.default_variant
            allow = settings.auto_download_models if download is None else bool(download)
            manager.ensure(info, info.variant(variant), download=allow, cancel=cancel)     # raises with the hint
        checkpoint = manager.path(info.id, info.variant(variant).weights[0])
        return Sam2ImageBackend(variant, checkpoint, device)
    raise UnknownModelError(f"Unknown subject-selection backend '{name}'.")


# --------------------------------------------------------------------------- SAM 2 on one image


class Sam2ImageBackend:
    """Box and / or clicks -> precise mask on a single frame (SAM 2.1 image predictor)."""

    name = "sam2"

    def __init__(self, variant: str, checkpoint: Any, device: Any) -> None:
        import torch

        from ..models.sam2.streaming import build_predictor

        self.variant = variant
        self.device = torch.device(device)
        try:
            self._model = build_predictor(variant, checkpoint, self.device)
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            self._predictor = SAM2ImagePredictor(self._model)
        except Exception as exc:
            raise ModelLoadError(f"Could not load SAM 2.1 ({variant}) for subject selection: {exc}") from exc
        self._dtype = None
        if self.device.type == "cuda":
            self._dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def _amp(self) -> Any:
        import torch

        return torch.autocast("cuda", dtype=self._dtype) if self._dtype is not None else contextlib.nullcontext()

    def segment(self, frame_rgb: np.ndarray, box: Optional[Sequence[float]] = None,
                points: Sequence[Point] = ()) -> Optional[ResolvedSubject]:
        import torch

        if box is None and not points:
            return None
        height, width = frame_rgb.shape[:2]
        coords = labels = None
        if points:
            coords = np.asarray([[x, y] for x, y, _ in points], dtype=np.float32)
            labels = np.asarray([label for _, _, label in points], dtype=np.int32)
        box_array = None if box is None else np.asarray(clamp_box(box, width, height), dtype=np.float32)
        # Clicks alone are ambiguous (a part or the whole subject?): let SAM propose three masks and
        # take the best-rated one that honours every click - contains the positive ones, excludes the
        # negative ones.  A box is unambiguous: one mask.
        multimask = box is None
        with torch.inference_mode(), self._amp():
            self._predictor.set_image(np.ascontiguousarray(frame_rgb))
            masks, scores, _ = self._predictor.predict(
                point_coords=coords, point_labels=labels, box=box_array, multimask_output=multimask, return_logits=True,
            )
            self._predictor.reset_predictor()
        order = [int(i) for i in np.argsort(-np.asarray(scores))]
        best = order[0]
        if multimask and points:
            def honours(index: int) -> bool:
                candidate = np.asarray(masks[index]) > 0.0
                for x, y, label in points:
                    xi, yi = min(width - 1, max(0, int(round(x)))), min(height - 1, max(0, int(round(y))))
                    if bool(candidate[yi, xi]) != bool(label):
                        return False
                return True

            # SAM rates the smallest plausible part highest (the shirt under the cursor: 0.67) and
            # the whole person lowest (0.29) although that mask is the accurate one for the subject.
            # A click in a matting tool means "this subject", so the largest consistent candidate
            # wins unless it is implausible (nearly the whole frame, or rated as noise).
            consistent = [i for i in order if honours(i)]
            plausible = [i for i in consistent
                         if float(scores[i]) >= 0.15 and float((np.asarray(masks[i]) > 0.0).mean()) <= 0.85]
            if plausible:
                best = max(plausible, key=lambda i: float((np.asarray(masks[i]) > 0.0).sum()))
            elif consistent:
                best = consistent[0]
        logits = np.asarray(masks[best], dtype=np.float32)
        mask = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        if float((mask > 0.5).mean()) <= 0.0:
            return None
        kinds = [kind for kind, present in (("box", box is not None), ("points", bool(points))) if present]
        return ResolvedSubject(
            mask=mask.astype(np.float32), box=box_from_mask(mask), points=[(float(x), float(y), int(l)) for x, y, l in points],
            source="sam2:" + "+".join(kinds), score=float(scores[best]),
        )

    def close(self) -> None:
        self._predictor = None
        self._model = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - best effort
            pass


# --------------------------------------------------------------------------- automatic detection

_COCO = (
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)
_ANIMALS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
_VEHICLES = {"bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"}


def _class_prior(label: str) -> float:
    """How likely a detected class is to be *the* subject of a shot."""

    if label == "person":
        return 1.0
    if label in _ANIMALS:
        return 0.7
    if label in _VEHICLES:
        return 0.45
    return 0.35


class DetectorBackend:
    """Mask R-CNN instance detection with a primary-subject scoring rule."""

    name = "maskrcnn"
    MAX_SIDE = 1024              # detection runs on a copy this large at most (4K frames would cost gigabytes)
    MIN_SCORE = 0.5
    MIN_AREA = 0.005             # fraction of the frame

    def __init__(self, weights_path: Any) -> None:
        import torch
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

        self.device = torch.device("cpu")
        try:
            # weights=None / weights_backbone=None: torchvision must never download anything.
            model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, min_size=480, max_size=864,
                                             box_score_thresh=self.MIN_SCORE, box_detections_per_img=20)
            state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
            model.load_state_dict(state)
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load the subject detector from {weights_path}: {exc}",
                hint="Verify the file with 'python main.py models verify detector' or download it again.",
            ) from exc
        self._model = model.eval()

    def _detect(self, frame_rgb: np.ndarray) -> List[Dict[str, Any]]:
        import torch

        height, width = frame_rgb.shape[:2]
        scale = min(1.0, self.MAX_SIDE / float(max(height, width)))
        small = frame_rgb if scale == 1.0 else cv2.resize(
            frame_rgb, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(np.ascontiguousarray(small)).permute(2, 0, 1).float().div_(255.0)
        with torch.inference_mode():
            result = self._model([tensor])[0]
        instances: List[Dict[str, Any]] = []
        for box, label, score, mask in zip(result["boxes"], result["labels"], result["scores"], result["masks"]):
            if float(score) < self.MIN_SCORE:
                continue
            probability = mask[0].numpy()
            if scale != 1.0:
                probability = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
            area = float((probability > 0.5).mean())
            if area < self.MIN_AREA:
                continue
            index = int(label)
            instances.append({
                "mask": probability.astype(np.float32),
                "box": tuple(float(v) / scale for v in box.tolist()),
                "label": _COCO[index] if 0 <= index < len(_COCO) else str(index),
                "confidence": float(score),
                "area": area,
            })
        return instances

    @staticmethod
    def _score(instance: Dict[str, Any], width: int, height: int) -> float:
        """prior(class) x confidence x sqrt(area) x centrality (subjects sit near the centre, slightly high)."""

        ys, xs = np.nonzero(instance["mask"] > 0.5)
        cx, cy = (float(xs.mean()) / width, float(ys.mean()) / height) if xs.size else (0.5, 0.5)
        centrality = math.exp(-0.5 * (((cx - 0.5) / 0.35) ** 2 + ((cy - 0.45) / 0.35) ** 2))
        return _class_prior(instance["label"]) * instance["confidence"] * math.sqrt(instance["area"]) * (0.35 + 0.65 * centrality)

    def _subject(self, instance: Dict[str, Any], source: str, score: float) -> ResolvedSubject:
        mask = instance["mask"]
        return ResolvedSubject(mask=mask, box=box_from_mask(mask) or instance["box"], source=source, score=score,
                               label=instance["label"])

    def primary_subject(self, frame_rgb: np.ndarray) -> Optional[ResolvedSubject]:
        height, width = frame_rgb.shape[:2]
        instances = self._detect(frame_rgb)
        if not instances:
            return None
        ranked = sorted(((self._score(i, width, height), i) for i in instances), key=lambda item: -item[0])
        best_score, best = ranked[0]
        subject = self._subject(best, "auto:maskrcnn", float(best["confidence"]))
        if len(ranked) > 1 and ranked[1][0] >= 0.8 * best_score:
            other = ranked[1][1]
            subject.notes.append(
                f"Automatic selection was close: a {best['label']} was chosen, but another {other['label']} scored "
                "almost as high. Select the subject manually if the wrong one was picked."
            )
        return subject

    def match_prompt(self, frame_rgb: np.ndarray, box: Optional[Sequence[float]] = None,
                     points: Sequence[Point] = ()) -> Optional[ResolvedSubject]:
        """The detected instance that fits a user box best / contains the positive clicks and no negative one."""

        instances = self._detect(frame_rgb)
        if not instances:
            return None
        height, width = frame_rgb.shape[:2]
        best: Optional[Tuple[float, Dict[str, Any]]] = None
        # Without a box several instances can honour the clicks (a person and the bag they carry):
        # the likelier subject wins.
        instances = sorted(instances, key=lambda item: -_class_prior(item["label"]) * item["confidence"])
        for instance in instances:
            mask = instance["mask"] > 0.5
            fit = 1.0
            if box is not None:
                x1, y1, x2, y2 = clamp_box(box, width, height)
                inside = np.zeros_like(mask)
                inside[int(y1): int(math.ceil(y2)), int(x1): int(math.ceil(x2))] = True
                union = float((mask | inside).sum())
                fit = float((mask & inside).sum()) / union if union else 0.0
                if fit < 0.2:
                    continue
            ok = True
            for x, y, label in points:
                xi, yi = min(width - 1, max(0, int(round(x)))), min(height - 1, max(0, int(round(y))))
                if bool(mask[yi, xi]) != bool(label):
                    ok = False
                    break
            if not ok:
                continue
            if best is None or fit > best[0]:
                best = (fit, instance)
        if best is None:
            return None
        subject = self._subject(best[1], "maskrcnn:prompt", float(best[1]["confidence"]))
        subject.points = [(float(x), float(y), int(l)) for x, y, l in points]
        return subject

    def close(self) -> None:
        self._model = None


# --------------------------------------------------------------------------- text -> box


class TextGroundingBackend:
    """A text description -> the best matching box (Grounding DINO tiny, local files only)."""

    name = "grounding-dino"
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25
    MAX_SIDE = 800

    def __init__(self, directory: Any) -> None:
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            # local_files_only: the files were verified by the weights manager; nothing may be fetched.
            self._processor = AutoProcessor.from_pretrained(str(directory), local_files_only=True)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(str(directory), local_files_only=True).eval()
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load the text-grounding model from {directory}: {exc}",
                hint="Verify the files with 'python main.py models verify text_grounding'.",
            ) from exc

    def detect(self, frame_rgb: np.ndarray, text: str) -> Optional[Tuple[Box, float]]:
        import torch
        from PIL import Image

        phrase = " ".join(str(text).lower().split()).rstrip(".")
        if not phrase:
            return None
        height, width = frame_rgb.shape[:2]
        scale = min(1.0, self.MAX_SIDE / float(max(height, width)))
        image = Image.fromarray(np.ascontiguousarray(frame_rgb))
        if scale != 1.0:
            image = image.resize((max(1, int(round(width * scale))), max(1, int(round(height * scale)))), Image.BILINEAR)
        with torch.inference_mode():
            inputs = self._processor(images=image, text=phrase + ".", return_tensors="pt")
            outputs = self._model(**inputs)
            results = self._processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=self.BOX_THRESHOLD, text_threshold=self.TEXT_THRESHOLD,
                target_sizes=[(height, width)],
            )[0]
        if len(results["scores"]) == 0:
            return None
        best = int(torch.argmax(results["scores"]))
        box = clamp_box([float(v) for v in results["boxes"][best].tolist()], width, height)
        if (box[2] - box[0]) < 2 or (box[3] - box[1]) < 2:
            return None
        return box, float(results["scores"][best])

    def close(self) -> None:
        self._processor = None
        self._model = None
