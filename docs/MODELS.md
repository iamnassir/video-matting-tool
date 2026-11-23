# Models

The tool never bundles or silently downloads model weights. Every file comes from the official
source of its authors, is pinned by size and SHA-256, and is downloaded only when you ask for it:

```
python main.py models list                    # what is installed, licences
python main.py models info sam2               # variants, files, limitations
python main.py models download rvm --all --accept-license   # official source, verified
python main.py models verify                  # re-check every SHA-256
python main.py models remove matanyone
```

`--accept-license` is required for models whose licence has conditions (`rvm`, `detector`,
`matanyone`); without it the command prints the conditions and stops. `--all` fetches every
variant of a model.

Weights are stored in `models/<model id>/`. Mirrors or pinned copies can be configured per file with
`weight_overrides` in `configs/default.yaml`.

## Which model for which job

| `--model`   | What it does | Subject selection | Edges | Licence | Speed on RTX 3060 (measured) |
|-------------|--------------|-------------------|-------|---------|------------------------------|
| `rvm`       | Mattes **every person** in the shot | none of its own; the largest / selected person is isolated by connected-component tracking | true soft alpha | GPL-3.0 (conditional) | 80-117 fps inference, 20-33 fps end to end |
| `sam2`      | Follows **any one object** you select | box, clicks, mask, text, automatic | mask, softened by the refinement stage | Apache-2.0 (commercial use allowed) | about 8 fps |
| `sam2_rvm`  | Follows **one selected person** and gives a soft matte | as `sam2` | true soft alpha, gated by the SAM 2 track | GPL-3.0 + Apache-2.0 (conditional) | 6-7.5 fps |
| `matanyone` | Follows **one selected person**, best hair and motion blur | mask on the reference frame (boxes, clicks, text and auto are converted with the helper models) | true soft alpha | NTU S-Lab 1.0 (**non-commercial only**) | 10-12 fps at 720p |
| `sam2_matanyone` | Follows **one object with see-through parts** (a car with its windows) | mask on the reference frame with the openings cut out | steady outline, openings stay open | NTU S-Lab 1.0 + Apache-2.0 (**non-commercial only**) | 3 fps at 1080p; the whole clip is analysed before the first frame is written |

Rules of thumb:

* One person, clean shot, speed matters: `rvm`.
* Several people and you want one of them, commercial work: `sam2_rvm`.
* A subject that is not a person (product, animal, vehicle): `sam2`.
* Best possible matte for a person, non-commercial work: `matanyone`.
* A subject with see-through parts that must become background (a car with its windows, a person
  inside a vehicle): `sam2_matanyone` with a mask image that has those parts cut out. Measured on
  a 2046x1080 side shot of a moving car with driver (241 frames, a van overtaking behind it, a
  white roof against a white sky):
  * `sam2` alone kept the car but closed the windows again, and for 20 frames took the van as
    part of the car;
  * `matanyone` alone kept the windows open, but its roof edge wandered from frame to frame, it
    lost the windscreen opening in the first half of the clip and it kept passing vehicles;
  * `sam2_matanyone` gave one steady outline for all 241 frames, open windows and no vehicle
    outside the car. What passes *behind a window* (the van seen through the windscreen) can still
    show for some frames.

## Licences and commercial use

Licences are part of each adapter's metadata and are enforced, not just documented:

* `--commercial` (or `commercial_use: true` in the settings, or `VMT_COMMERCIAL_USE=1`) makes the
  tool **refuse** models whose licence forbids commercial use (`matanyone`, `sam2_matanyone`) with
  exit code 1.
* Models with conditions (`rvm`, `sam2_rvm`, the detector) run, and the condition is written into
  the processing report as a warning.

| Model | Code | Weights | Note |
|-------|------|---------|------|
| RVM | GPL-3.0 | official GitHub release, no separate statement | Processing your own footage is permitted; distributing the model or a product that bundles it falls under the GPL. |
| SAM 2.1 | Apache-2.0 | Apache-2.0 | Commercial use allowed. |
| MatAnyone / MatAnyone 2 | NTU S-Lab License 1.0 | same | Research and non-commercial use only; commercial use needs written permission from the authors. |
| Subject detector (Mask R-CNN, torchvision) | BSD-3-Clause | trained on COCO, no explicit licence statement | Review before commercial deployment. |
| Text grounding (Grounding DINO tiny) | Apache-2.0 | Apache-2.0 | |

This is engineering documentation, not legal advice: read the licence texts linked by
`python main.py models info <id>` before you deploy commercially.

## Robust Video Matting (`rvm`)

* Source: github.com/PeterL1n/RobustVideoMatting, official TorchScript files (MobileNetV3 and ResNet50).
* Install: nothing beyond the base requirements. `python main.py models download rvm --all --accept-license`.
* Limitations: trained on people only; mattes every person, so two people who touch cannot be
  separated (use `sam2_rvm`); hand-held objects are often dropped; no prompt.

## SAM 2.1 (`sam2`)

* Source: github.com/facebookresearch/sam2 at a pinned commit; checkpoints from
  `dl.fbaipublicfiles.com/segment_anything_2/092824/`. Variants: tiny, small (default), base_plus, large.
* Install: `python scripts/install_models.py sam2`, then `python main.py models download sam2`.
* The adapter streams frames through the video predictor with a bounded memory window, so long
  videos do not accumulate memory (about 0.7 GB of VRAM with the small variant).
* Limitations: produces a segmentation mask, not an alpha matte. The refinement stage converts the
  mask to a soft matte, but fine hair is approximated; one frame per step (no batching).

## SAM 2 + RVM hybrid (`sam2_rvm`)

Runs both networks on every frame: `alpha = RVM alpha x (SAM 2 mask, grown and softened)`, and parts
of the selected subject that RVM missed fall back to the SAM 2 mask (`-O fill_from_mask=false`
switches that off). Needs the `sam2` package and the weights of both models. People only.

## SAM 2 + MatAnyone for objects with openings (`sam2_matanyone`)

For a rigid thing you can see through in places. Needs the packages and weights of `sam2` and
`matanyone` (nothing else to download) and a mask on the reference frame in which the openings
are cut out; a still of the wanted result on flat green works as that mask.

How it works:

* SAM 2 follows the *solid* outline of the selection (every opening filled), MatAnyone the
  selection as it is. The result is MatAnyone's matte inside SAM 2's outline: the outline from the
  model that sees objects, the openings from the model that decides pixel by pixel.
* In a rim along the outline (`-O rim_depth=44`, pixels at 1080p) everything is made opaque,
  except an opening that reaches deeper than the rim. That repairs the pieces MatAnyone drops
  where the object has the colour of the background (a white roof against a white sky).
  `-O restore_rim=false` switches it off.
* **One outline for the whole clip.** A rigid object does not change its shape, it only moves.
  The tool measures that motion on every frame (corner features on the object, affine fit,
  re-measured against the reference frame so that errors do not add up) and, when the clip has
  been seen completely, lays all SAM 2 outlines on top of each other in the object's own
  coordinates and takes the majority. A van that is behind the car for twenty frames out of two
  hundred is outvoted; parts of the object that enter the picture later are covered too. Inside
  the reference frame the outline you gave is used as it is.
* This needs the whole clip, so the model runs over every frame first and rendering starts
  afterwards (a two-pass model: an interrupted run starts again instead of resuming).

Options (`-O key=value`): `rigid_outline=auto|on|off` (auto uses the one outline when it agrees
with the frame-by-frame outlines to 93 % or more, `rigid_agreement`; the report states the
agreement under `model.runtime.rigid_outline`), `outline_choke=2` (pixels by which that outline
is drawn in), `gate_margin=4`, `internal_short_side` (passed on to MatAnyone).

Limitations: the outline is hard (no hair detail: use `matanyone` or `sam2_rvm` for people);
something that passes behind an opening is judged by MatAnyone alone and can stay visible for
some frames; a subject that changes its shape gets frame-by-frame outlines, which can still be
fooled by a similar object directly behind it; non-commercial licence.

## MatAnyone / MatAnyone 2 (`matanyone`)

* Source: github.com/pq-yang/MatAnyone (pinned commit) and the official v1.0.0 release files of
  MatAnyone and MatAnyone 2. Default variant: `matanyone2`.
* Install: `python scripts/install_models.py matanyone --accept-license`, then
  `python main.py models download matanyone --accept-license`.
* Needs a mask on the reference frame. `--subject mask:<file>` uses yours; box, click, text and
  automatic selection are turned into that mask by SAM 2 and the detector, which must then be installed.
* Memory grows with the square of the frame size, so frames are analysed at a reduced internal
  resolution chosen from the GPU memory that is really free (short side 432-1080: 720 needs about
  1.3 GB beyond the weights and is chosen from 2.1 GB free) and the matte is returned at full size.
  The report states the size used (`model.runtime.internal_short_side`); the edge of the matte is
  only as fine as this size. `-O internal_short_side=720` forces a value. On out-of-memory the
  adapter steps down and continues from its last matte.
* The network is built without Hydra and without any download; the upstream default would fetch
  ImageNet backbones into the user profile.

## Helper models (subject selection)

| Id | Used for | Install |
|----|----------|---------|
| `detector` | `--subject auto`, and turning a click on a garment into "the whole person" | `python main.py models download detector --accept-license` |
| `text_grounding` | `--subject "text:the woman on the left"` | `python scripts/install_models.py text`, then `python main.py models download text_grounding` |
| SAM 2 image predictor | box / clicks to a precise mask | comes with `sam2` |

Helpers are loaded for the subject stage and unloaded before the matting model starts. The detector
and the text model run on the CPU (about 10-25 s once per job) so they never compete for VRAM.

Known limitation: text grounding returns its best box even for a phrase that matches nothing in the
picture. Check the selection with "Preview selection" in the web UI.

## SAM2Matting: evaluated, not integrated

The specification names "SAM2Matting" as a candidate. The public project of that name
(FudanCVL/SAM2Matting) was evaluated and deliberately not integrated:

* its licence is CC BY-NC (non-commercial), so it adds no commercial option;
* it pins PyTorch 2.8 and ships a modified copy of the `sam2` package under the same import name,
  which cannot coexist with the official SAM 2 package this tool uses;
* it could not be validated on the available hardware next to the other models.

The same use case, "SAM 2 selects, a matting network draws the edges", is covered by `sam2_rvm`
(commercially usable) and by `matanyone` (highest quality). It can be added later as a plugin without
touching the core.

## Adding a model (plugin)

An adapter is one class. Put it in any importable module and list the module under `plugins:` in the
settings file:

```python
from app.models.base import (LicenseInfo, MatteOutput, MattingModel, ModelCapabilities,
                             ModelInfo, ModelVariant, WeightFile)
from app.models.registry import register_model

@register_model
class MyModel(MattingModel):
    info = ModelInfo(
        id="mymodel", name="My model", description="...",
        license=LicenseInfo(name="MIT", url="https://...", commercial_use="allowed"),
        capabilities=ModelCapabilities(alpha_type="alpha", temporal="recurrent"),
        variants=(ModelVariant(id="base", label="Base", weights=(WeightFile(
            filename="mymodel.pth", url="https://...", size_bytes=123, sha256="..."),)),),
        default_variant="base",
        limitations=("...",),
    )

    def load(self):                       # weights -> self.device
        self.net = ...; self.loaded = True

    def start_sequence(self, seq):        # reset temporal state; seq.subject carries the prompt
        ...

    def process(self, frames, frame_indices):   # frames: [B,3,H,W] float 0..1, display orientation
        return MatteOutput(alpha=self.net(frames))   # [B,1,H,W] float32 0..1

    def reduce_memory(self):              # optional: cheaper configuration after out-of-memory
        return False
```

The pipeline provides everything else: decoding, batching with out-of-memory recovery, the matte
cache, refinement, compositing, encoding, audio, verification and the report. The contract is
described in `app/models/base.py`.
