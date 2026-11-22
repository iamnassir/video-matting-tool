"""Native-domain compositor: matte + source planes -> output planes for every deliverable.

The compositor works directly on the planes of the source's working format and only detours
through R'G'B' where it has to (the semi-transparent band).  For a background colour ``Bg``,
source colour ``I`` and matte ``a`` a *composite* target is built like this:

1. ``F`` = foreground colour: ``I`` where the matte is 1, the Blur-Fusion estimate inside the band
   (scaled by ``decontaminate``) so the old background leaves no halo.
2. ``C = a*F + (1-a)*Bg`` in float R'G'B', converted to the target's planes (matrix and range of the
   source, the target's bit depth and subsampling).
3. **Opaque core restore** - evaluated per plane, at that plane's resolution: wherever every matte
   sample in a plane sample's footprint is exactly 1, the output is the source sample, taken from
   the source planes without any colour matrix (invariant I4: bit-exact when source and target
   have the same format).
4. **Background constant** - wherever every matte sample in the footprint is exactly 0, the output
   is the pre-computed integer code value of ``Bg`` for that plane (invariant I5: one constant per
   plane, no gradients, no noise).

Steps 3 and 4 are *overrides* of the float result, so the two invariants do not depend on any
floating-point behaviour.  Transparent targets, matte targets, RGBA for PNG export and the preview
atlas are produced from the same per-batch context, which caches everything they share.

Rules that hold for every target
--------------------------------
* **A frame renders the same bits whatever it is batched with.**  Every decision (is there a band,
  is a sample core / clear) is taken per frame, and the arithmetic uses only element-wise kernels
  that round identically at every buffer position (see "Determinism" in
  :mod:`app.compositing.colors`).  A resumed run therefore reproduces the interrupted one exactly.
* **Fully transparent samples of transparent targets are black**, in straight and premultiplied
  mode alike: a sample without coverage has no colour, a constant costs the encoder nothing, and
  the removed background can never be recovered from the colour planes.
* **Colour next to transparency is always handled alpha-weighted** in straight mode: subsampled
  chroma is averaged with the matte as weight, and a resized picture is resampled premultiplied
  and divided afterwards.  Both keep whatever lies under invisible pixels out of visible ones.
* **Bounded working memory.**  A target is rendered a few frames at a time (about
  ``_CHUNK_PIXELS`` pixels per step) and the integer results are concatenated, so the float
  temporaries do not grow with the batch size the model happens to use.
* Orientation: everything is in coded orientation except :func:`preview_atlas`, which receives
  display-oriented tensors.  Nothing modifies its inputs; all tensors are created on the device of
  the inputs.  CUDA out-of-memory errors are deliberately not wrapped, so the pipeline's batch
  reduction (``app.core.errors.is_cuda_oom``) still recognises them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..core.errors import ConfigError
from ..core.pixfmt import PixFmt, ResolvedColor, make_pixfmt, quantization
from ..refinement.foreground import estimate_foreground
from ..utils.torch_ops import luma as bt709_luma
from ..utils.torch_ops import resize
from .colors import ColorConverter, convert_planes, downsample_chroma, to_integer_planes

log = logging.getLogger(__name__)

__all__ = ["CompositeContext", "Compositor", "Target", "preview_atlas", "rgb8_to_yuv420p"]

_KINDS = ("composite", "alpha", "matte")
_ALPHA_MODES = ("straight", "premultiplied")
_BLACK: Tuple[int, int, int] = (0, 0, 0)

#: A target is rendered in steps of about this many pixels (two 1080p frames, one 2160p frame).
#: Rendering needs roughly a dozen float32 planes per frame; stepping keeps that transient memory
#: independent of the batch size.  The result does not depend on the step (frames never interact).
_CHUNK_PIXELS = 1 << 22

#: Brightness of everything that is not the subject in the "segmented" preview tile.
_PREVIEW_DIM = 0.35
#: 8-bit R'G'B' of the outline along the alpha = 0.5 contour in the "segmented" preview tile.
_PREVIEW_OUTLINE = (255, 235, 40)
#: The two greys of the transparency checkerboard in the preview.
_CHECKER_GREYS = (0.42, 0.62)


# --------------------------------------------------------------------------- targets


@dataclass(frozen=True)
class Target:
    """One deliverable rendered from a batch.

    ``kind``:

    * ``"composite"`` - the subject over the constant ``background`` (8-bit R'G'B', required).
    * ``"alpha"`` - the foreground colour in the colour planes and the matte in the alpha plane
      (``fmt.has_alpha`` must be true).  ``alpha_mode`` selects straight or premultiplied colour.
    * ``"matte"`` - the matte as a grey-scale picture (white = subject).

    ``size`` is the ``(width, height)`` of the output; ``None`` means the source size, i.e. no
    scaling, which is the only mode in which the pixel-preservation invariant applies.

    The instance is validated on construction (``ConfigError``), hashable, and therefore usable as
    a cache key.
    """

    name: str
    kind: str
    fmt: PixFmt
    background: Optional[Tuple[int, int, int]] = None
    size: Optional[Tuple[int, int]] = None
    alpha_mode: str = "straight"

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ConfigError(
                f"Target {self.name!r}: unknown kind {self.kind!r}.", hint=f"Use one of: {', '.join(_KINDS)}."
            )
        if self.alpha_mode not in _ALPHA_MODES:
            raise ConfigError(
                f"Target {self.name!r}: unknown alpha mode {self.alpha_mode!r}.",
                hint=f"Use one of: {', '.join(_ALPHA_MODES)}.",
            )
        if self.background is not None:
            try:
                colour = tuple(int(v) for v in self.background)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"Target {self.name!r}: background must be three 8-bit R'G'B' values, got {self.background!r}."
                ) from exc
            if len(colour) != 3 or not all(0 <= v <= 255 for v in colour):
                raise ConfigError(
                    f"Target {self.name!r}: background must be three values in 0..255, got {self.background!r}."
                )
            object.__setattr__(self, "background", colour)
        if self.kind == "composite" and self.background is None:
            raise ConfigError(
                f"Target {self.name!r}: a composite needs a background colour.",
                hint="Pass background=(R, G, B), or use kind='alpha' for a transparent output.",
            )
        if self.kind == "alpha" and not self.fmt.has_alpha:
            raise ConfigError(
                f"Target {self.name!r}: pixel format {self.fmt.name} has no alpha plane.",
                hint="Use a format with alpha (yuva444p10le, yuva420p, gbrap ...) for a transparent output.",
            )
        if self.size is not None:
            try:
                size = tuple(int(v) for v in self.size)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"Target {self.name!r}: size must be (width, height), got {self.size!r}.") from exc
            if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
                raise ConfigError(f"Target {self.name!r}: size must be a positive (width, height), got {self.size!r}.")
            object.__setattr__(self, "size", size)


@dataclass(frozen=True)
class _Plan:
    """Geometry of a target: where the (possibly scaled) picture sits inside the output frame."""

    scaled: bool
    out_w: int
    out_h: int
    in_w: int       # size of the picture after scaling
    in_h: int
    off_x: int      # its top-left corner inside the output frame
    off_y: int


# --------------------------------------------------------------------------- per-batch context


class CompositeContext:
    """State shared by all targets rendered from one batch (created by :meth:`Compositor.begin`).

    Everything expensive that several targets need is computed lazily and at most once: the
    R'G'B' view of the source, the foreground estimate and the per-plane core / clear masks.  A
    batch that feeds several deliverables (main output, ProRes 4444, matte, PNG ...) therefore pays
    for each of them only once.
    """

    def __init__(
        self,
        compositor: "Compositor",
        planes: List[torch.Tensor],
        alpha: torch.Tensor,
        rgb: Optional[torch.Tensor],
    ) -> None:
        self._compositor = compositor
        #: float32 colour planes of the source (code values, coded orientation); never modified
        self.planes = planes
        #: sanitised matte, float32 [B, 1, H, W] in 0..1
        self.alpha = alpha
        self._rgb = rgb
        self._foreground: Optional[torch.Tensor] = None
        self._band_frames: Optional[torch.Tensor] = None
        self._masks: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        low, high = torch.aminmax(alpha.flatten(1), dim=1)
        #: per frame: every matte sample is exactly 1 / exactly 0 (such frames need no blending)
        self.frame_opaque: List[bool] = (low >= 1.0).tolist()
        self.frame_clear: List[bool] = (high <= 0.0).tolist()

    # -- basic facts ---------------------------------------------------------------------
    @property
    def batch(self) -> int:
        return int(self.alpha.shape[0])

    @property
    def device(self) -> torch.device:
        return self.alpha.device

    @property
    def all_opaque(self) -> bool:
        """Every matte sample of the batch is exactly 1."""

        return all(self.frame_opaque)

    @property
    def all_clear(self) -> bool:
        """Every matte sample of the batch is exactly 0."""

        return all(self.frame_clear)

    # -- lazily derived tensors ----------------------------------------------------------
    @property
    def rgb(self) -> torch.Tensor:
        """R'G'B' of the source, ``float32 [B, 3, H, W]`` (given to ``begin`` or derived on demand)."""

        if self._rgb is None:
            self._rgb = self._compositor.converter.planes_to_rgb(self.planes)
        return self._rgb

    @property
    def band_frames(self) -> torch.Tensor:
        """``bool [B]``: True for frames that contain at least one sample with ``0 < alpha < 1``."""

        if self._band_frames is None:
            core, clear = self.masks(0, 0)                      # shared with the luma overrides
            self._band_frames = (~(core | clear)).flatten(1).any(dim=1)
        return self._band_frames

    @property
    def foreground(self) -> torch.Tensor:
        """Foreground colour ``F``, ``float32 [B, 3, H, W]``; equals ``rgb`` wherever alpha is 1.

        The estimate is only computed for frames that actually have a semi-transparent band (and
        only when decontamination is enabled); every other frame uses ``rgb`` as is.  The decision
        is made per frame, never per batch, so a frame renders identically whatever it is batched
        with.  May be the ``rgb`` tensor itself: do not modify it.
        """

        if self._foreground is None:
            comp = self._compositor
            rgb = self.rgb
            foreground = rgb
            if comp.decontaminate > 0.0 and not (self.all_opaque or self.all_clear):
                flags = self.band_frames
                count = int(flags.sum())
                if count == self.batch:
                    foreground = comp._estimate(rgb, self.alpha)
                elif count > 0:
                    index = torch.nonzero(flags).flatten()
                    foreground = rgb.clone()
                    foreground[index] = comp._estimate(rgb[index], self.alpha[index])
            self._foreground = foreground
        return self._foreground

    def masks(self, chroma_w: int = 0, chroma_h: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(core, clear)`` boolean masks for planes with the given log2 subsampling.

        ``core`` is True where *every* matte sample in the plane sample's footprint is exactly 1
        (min-pooling), ``clear`` where every one is exactly 0 (max-pooling).  ``ceil_mode`` keeps the
        partial footprints of odd frame sizes, and pooling ignores samples outside the frame.
        """

        key = (int(chroma_w), int(chroma_h))
        cached = self._masks.get(key)
        if cached is None:
            a = self.alpha
            if key == (0, 0):
                low = high = a
            else:
                kernel = (1 << key[1], 1 << key[0])
                high = F.max_pool2d(a, kernel_size=kernel, stride=kernel, ceil_mode=True)
                low = -F.max_pool2d(-a, kernel_size=kernel, stride=kernel, ceil_mode=True)
            cached = (low >= 1.0, high <= 0.0)
            self._masks[key] = cached
        return cached

    def core_planes(self, fmt: PixFmt, start: int = 0, stop: Optional[int] = None) -> List[torch.Tensor]:
        """Source samples of frames ``start:stop`` in the colour planes of *fmt* (what the opaque
        core receives).

        Same sample kind on both sides (Y'CbCr / gray, or RGB): a matrix-free plane conversion -
        the identity when the formats match, hence bit-exact.  Across kinds (RGB source with a
        Y'CbCr target or the reverse) the samples have to go through the colour matrix; that case
        is necessarily **not** bit-exact.  Do not modify the returned tensors: they may be views of
        the source planes.
        """

        comp = self._compositor
        frames = slice(start, self.batch if stop is None else stop)
        src = comp.src_fmt
        if (src.family == "rgb") == (fmt.family == "rgb"):
            return convert_planes([p[frames] for p in self.planes], src, fmt, comp.color.full_range)
        return comp.converter.rgb_to_planes(self.rgb[frames], fmt)


# --------------------------------------------------------------------------- compositor


class Compositor:
    """Renders the deliverables of one stream.

    ``width`` / ``height`` are the source frame size in coded orientation.  ``decontaminate`` is the
    strength of the foreground colour estimation in the edge band (0 = keep the observed colours,
    1 = full estimate); the two radii are box-blur radii in pixels **at the actual frame size**
    (a radius ``r`` is a ``(2r + 1)``-wide window, see
    :func:`app.refinement.foreground.estimate_foreground`).

    Usage per batch::

        ctx = compositor.begin(planes, alpha, rgb)        # rgb optional (already computed for the model)
        for target in targets:
            integer_planes = compositor.render(ctx, target)
    """

    def __init__(
        self,
        src_fmt: PixFmt,
        color: ResolvedColor,
        width: int,
        height: int,
        *,
        decontaminate: float = 1.0,
        decontaminate_radius: float = 90.0,
        decontaminate_refine_radius: float = 6.0,
    ) -> None:
        if int(width) <= 0 or int(height) <= 0:
            raise ConfigError(f"Invalid frame size {width}x{height} for the compositor.")
        self.src_fmt = src_fmt
        self.color = color
        self.width = int(width)
        self.height = int(height)
        strength = float(decontaminate)
        self.decontaminate = min(max(strength, 0.0), 1.0) if strength == strength else 0.0
        self.decontaminate_radius = max(0.0, float(decontaminate_radius))
        self.decontaminate_refine_radius = max(0.0, float(decontaminate_refine_radius))
        self.converter = ColorConverter(src_fmt, color)
        self._plans: Dict[Target, _Plan] = {}

    # -- batch entry ---------------------------------------------------------------------
    @torch.no_grad()
    def begin(
        self,
        planes: Sequence[torch.Tensor],
        alpha: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
    ) -> CompositeContext:
        """Start compositing one batch.

        ``planes``: source planes, code values, ``[B, 1, h, w]`` each, coded orientation (an alpha
        plane of the source is ignored).  ``alpha``: ``[B, 1, H, W]`` in 0..1, coded orientation.
        ``rgb``: optional ``planes_to_rgb(planes)`` if the caller already has it.

        The matte is copied and sanitised (clamped to 0..1, undefined samples become 0) so that
        "exactly 1" and "exactly 0" are well defined; the inputs themselves are left untouched.
        """

        fmt = self.src_fmt
        count = fmt.color_planes
        if len(planes) < count:
            raise ValueError(f"Compositor.begin: {fmt.name} needs {count} colour plane(s), got {len(planes)}")
        batch = int(planes[0].shape[0]) if planes[0].dim() == 4 else -1
        source: List[torch.Tensor] = []
        for index in range(count):
            plane = planes[index]
            expected = (batch, 1) + fmt.plane_size(index, self.width, self.height)
            if plane.dim() != 4 or tuple(plane.shape) != expected:
                raise ValueError(
                    f"Compositor.begin: plane {index} of {fmt.name} has shape {tuple(plane.shape)}, expected "
                    f"[B, 1, {expected[2]}, {expected[3]}] for a {self.width}x{self.height} frame"
                )
            source.append(plane if plane.dtype == torch.float32 else plane.to(torch.float32))
        if batch < 1:
            raise ValueError("Compositor.begin: the batch is empty")
        device = source[0].device
        if alpha.dim() != 4 or tuple(alpha.shape) != (batch, 1, self.height, self.width):
            raise ValueError(
                f"Compositor.begin: alpha has shape {tuple(alpha.shape)}, expected "
                f"({batch}, 1, {self.height}, {self.width}) in coded orientation"
            )
        matte = torch.nan_to_num(alpha.to(device=device, dtype=torch.float32), nan=0.0).clamp_(0.0, 1.0)
        if rgb is not None:
            if rgb.dim() != 4 or tuple(rgb.shape) != (batch, 3, self.height, self.width):
                raise ValueError(
                    f"Compositor.begin: rgb has shape {tuple(rgb.shape)}, expected "
                    f"({batch}, 3, {self.height}, {self.width}) in coded orientation"
                )
            rgb = rgb.to(device=device, dtype=torch.float32)
        return CompositeContext(self, source, matte, rgb)

    # -- rendering -----------------------------------------------------------------------
    @torch.no_grad()
    def render(self, ctx: CompositeContext, target: Target) -> List[torch.Tensor]:
        """Render *target* from the batch in *ctx*.

        Returns integer planes (see :func:`app.compositing.colors.to_integer_planes`) in
        ``target.fmt`` plane order, each ``[B, 1, h', w']``.

        Per kind:

        * ``"composite"`` - see the module docstring.  When source and target differ in family
          (RGB source, Y'CbCr target or the reverse) the opaque core has to pass the colour matrix
          and is therefore **not** bit-exact; the range of such a target is
          ``ColorConverter.full_range_for(target.fmt)``.
        * ``"alpha"`` - the colour planes carry ``F`` (straight) or ``a*F`` over black
          (premultiplied), the last plane ``round(a * max)`` in full range.  The opaque core holds
          the source samples in both modes; samples whose whole footprint is transparent hold the
          code values of black in both modes.
        * ``"matte"`` - Y'CbCr: limited-range luma (16..235 scaled to the bit depth) with neutral
          chroma; gray: full range ``0..max``; RGB: three identical planes ``a * max``.
        * A composite or matte target whose format has an alpha plane gets a fully opaque one.

        ``target.size`` different from the source size: the finished float planes are scaled per
        plane (bicubic, antialiased, clamped to the code range - a matte to black..white) to fit
        inside the target with the aspect ratio preserved, centred, and padded with the background
        constant (composite), black with alpha 0 (alpha) or matte black (matte).  Subsampled
        formats need even target dimensions (``ConfigError``).  A mismatch of less than two pixels
        - the rounding of an "<N>p" request - is absorbed by stretching instead of padding.  The
        pixel-preservation invariant does not apply to a resized target, except that pure padding
        (no size change of the picture) leaves the picture untouched.
        """

        self._check_context(ctx)
        plan = self._plan(target)
        parts = [self._render_frames(ctx, target, plan, start, stop) for start, stop in self._steps(ctx)]
        if len(parts) == 1:
            return parts[0]
        return [torch.cat([part[index] for part in parts], dim=0) for index in range(len(parts[0]))]

    @torch.no_grad()
    def render_rgba(self, ctx: CompositeContext, bit_depth: int = 8, alpha_mode: str = "straight") -> torch.Tensor:
        """``[B, H, W, 4]`` R, G, B, A for PNG export (coded orientation).

        ``uint8`` for 8 bit, ``int32`` holding 0..65535 for 16 bit.  Colours are the foreground
        estimate ``F`` (straight) or ``a*F`` (premultiplied); alpha is ``round(a * max)``.  Pixels
        whose stored alpha is 0 are transparent black in both modes.
        """

        self._check_context(ctx)
        if bit_depth not in (8, 16):
            raise ConfigError(f"PNG export supports 8 or 16 bits per channel, not {bit_depth}.")
        if alpha_mode not in _ALPHA_MODES:
            raise ConfigError(
                f"Unknown alpha mode {alpha_mode!r}.", hint=f"Use one of: {', '.join(_ALPHA_MODES)}."
            )
        fmt = make_pixfmt("rgb", bit_depth, has_alpha=True)
        peak = float(fmt.max_value)
        foreground = ctx.foreground
        parts: List[torch.Tensor] = []
        for start, stop in self._steps(ctx):
            alpha = ctx.alpha[start:stop]
            colour = foreground[start:stop]
            if alpha_mode == "premultiplied":
                colour = colour * alpha
            rgba = to_integer_planes([torch.cat([colour, alpha], dim=1).mul_(peak)], fmt)[0]
            if alpha_mode == "straight":
                rgba[:, 0:3].mul_(rgba[:, 3:4] > 0)
            parts.append(rgba.permute(0, 2, 3, 1).contiguous())
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    # -- internals -----------------------------------------------------------------------
    def _check_context(self, ctx: CompositeContext) -> None:
        if ctx._compositor is not self:
            raise ValueError("This CompositeContext was created by a different Compositor.")

    def _steps(self, ctx: CompositeContext) -> List[Tuple[int, int]]:
        """Frame ranges ``(start, stop)`` in which a batch is rendered (see ``_CHUNK_PIXELS``)."""

        step = max(1, _CHUNK_PIXELS // (self.width * self.height))
        return [(start, min(start + step, ctx.batch)) for start in range(0, ctx.batch, step)]

    def _estimate(self, rgb: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        return estimate_foreground(
            rgb,
            alpha,
            strength=self.decontaminate,
            radius=self.decontaminate_radius,
            refine_radius=self.decontaminate_refine_radius,
        )

    def _render_frames(
        self, ctx: CompositeContext, target: Target, plan: _Plan, start: int, stop: int
    ) -> List[torch.Tensor]:
        """Integer planes of *target* for the frames ``start:stop`` of the batch."""

        fmt = target.fmt
        frames = stop - start
        peak = float(fmt.max_value)
        code_range = (0.0, peak)

        if target.kind == "alpha":
            straight = target.alpha_mode == "straight"
            pads = [float(v) for v in self.converter.color_code_values(_BLACK, fmt)]
            # A resized straight picture is built premultiplied and divided after scaling.
            planes = self._colour_planes(ctx, fmt, _BLACK, pads, straight and not plan.scaled, start, stop)
            planes.append(ctx.alpha[start:stop] * peak)
            pads.append(0.0)
            if plan.scaled and straight:
                planes = self._fit_straight(planes, fmt, plan, pads)
            elif plan.scaled:
                planes = self._fit(planes, fmt, plan, pads, [code_range] * len(planes))
            return to_integer_planes(planes, fmt)

        if target.kind == "matte":
            planes, pads, limits = self._matte_planes(ctx, fmt, start, stop)
        else:
            pads = [float(v) for v in self.converter.color_code_values(target.background, fmt)]
            planes = self._colour_planes(ctx, fmt, target.background, pads, False, start, stop)
            limits = [code_range] * len(planes)
        if fmt.has_alpha:
            planes.append(
                torch.full((frames, 1, self.height, self.width), peak, dtype=torch.float32, device=ctx.device)
            )
            pads.append(peak)
            limits.append(code_range)
        if plan.scaled:
            planes = self._fit(planes, fmt, plan, pads, limits)
        return to_integer_planes(planes, fmt)

    def _colour_planes(
        self,
        ctx: CompositeContext,
        fmt: PixFmt,
        background: Tuple[int, int, int],
        constants: Sequence[float],
        straight: bool,
        start: int,
        stop: int,
    ) -> List[torch.Tensor]:
        """Float colour planes of *fmt* (source size, frames ``start:stop``) with both exact
        overrides applied.

        ``straight=False``: the subject over *background* (``a*F + (1-a)*Bg``).
        ``straight=True``: ``F`` itself; *background* is only what fully transparent samples get.
        ``constants`` are the code values of *background* per plane.
        """

        count = fmt.color_planes
        frames = slice(start, stop)
        if all(ctx.frame_clear[frames]):
            # No subject in these frames: nothing to blend, every sample is the constant.
            return [
                torch.full(
                    (stop - start, 1) + fmt.plane_size(index, self.width, self.height),
                    float(constants[index]), dtype=torch.float32, device=ctx.device,
                )
                for index in range(count)
            ]
        core = ctx.core_planes(fmt, start, stop)
        if all(ctx.frame_opaque[frames]):
            return list(core)

        alpha = ctx.alpha[frames]
        foreground = ctx.foreground[frames]
        if straight:
            # Subsampled chroma of straight colour is averaged with the matte as weight: a chroma
            # sample shared by visible and invisible pixels must carry the colour of the visible
            # ones only, whatever sits under the transparent ones.
            blended = self.converter.rgb_to_planes(foreground, fmt, chroma_weight=alpha)
        else:
            colour = torch.tensor(
                [c / 255.0 for c in background], dtype=torch.float32, device=ctx.device
            ).view(1, 3, 1, 1)
            # a*F + (1-a)*Bg with plain products: exact at both ends (Bg at a == 0, F at a == 1) and,
            # unlike torch.lerp, rounded identically wherever the frame sits in the batch.
            blended_rgb = foreground * alpha
            blended_rgb.add_((1.0 - alpha) * colour)
            blended = self.converter.rgb_to_planes(blended_rgb, fmt)
            del blended_rgb

        out: List[torch.Tensor] = []
        for index in range(count):
            sub = (fmt.chroma_w, fmt.chroma_h) if fmt.is_chroma_plane(index) else (0, 0)
            core_mask, clear_mask = ctx.masks(*sub)
            plane = torch.where(core_mask[frames], core[index], blended[index])
            out.append(torch.where(clear_mask[frames], float(constants[index]), plane))
        return out

    def _matte_planes(
        self, ctx: CompositeContext, fmt: PixFmt, start: int, stop: int
    ) -> Tuple[List[torch.Tensor], List[float], List[Tuple[float, float]]]:
        """Float planes of the grey-scale matte picture, the per-plane value of "matte black" and
        the per-plane ``(black, white)`` levels a resampled matte must stay within."""

        alpha = ctx.alpha[start:stop]
        peak = float(fmt.max_value)
        if fmt.family == "yuv":
            # A matte video is watched and keyed like any other video, so it uses video levels.
            luma_offset, luma_scale, neutral, _ = quantization(fmt, False)
            luma = (alpha * luma_scale).add_(luma_offset)
            chroma_shape = (stop - start, 1) + fmt.plane_size(1, self.width, self.height)
            chroma = torch.full(chroma_shape, neutral, dtype=torch.float32, device=ctx.device)
            levels = [(luma_offset, luma_offset + luma_scale), (neutral, neutral), (neutral, neutral)]
            return [luma, chroma, chroma], [luma_offset, neutral, neutral], levels
        grey = alpha * peak
        count = fmt.color_planes
        return [grey] * count, [0.0] * count, [(0.0, peak)] * count

    def _plan(self, target: Target) -> _Plan:
        plan = self._plans.get(target)
        if plan is None:
            plan = self._make_plan(target)
            self._plans[target] = plan
        return plan

    def _make_plan(self, target: Target) -> _Plan:
        src_w, src_h = self.width, self.height
        if target.size is None or tuple(target.size) == (src_w, src_h):
            return _Plan(False, src_w, src_h, src_w, src_h, 0, 0)
        out_w, out_h = target.size
        fmt = target.fmt
        if (fmt.chroma_w and out_w % 2) or (fmt.chroma_h and out_h % 2):
            raise ConfigError(
                f"Target {target.name!r}: output size {out_w}x{out_h} is not valid for {fmt.name}; "
                f"chroma-subsampled formats need even dimensions.",
                hint="Choose an even width / height, or a 4:4:4 pixel format.",
            )
        scale = min(out_w / float(src_w), out_h / float(src_h))
        fit_w, fit_h = src_w * scale, src_h * scale
        # A bar thinner than two pixels is rounding noise of the requested size, not letterboxing.
        in_w = out_w if out_w - fit_w < 2.0 else int(round(fit_w))
        in_h = out_h if out_h - fit_h < 2.0 else int(round(fit_h))
        # Subsampled chroma needs the picture on an even grid so every plane scales to whole samples.
        if fmt.chroma_w:
            in_w -= in_w % 2
        if fmt.chroma_h:
            in_h -= in_h % 2
        in_w = min(max(in_w, 2 if fmt.chroma_w else 1), out_w)
        in_h = min(max(in_h, 2 if fmt.chroma_h else 1), out_h)
        off_x = (out_w - in_w) // 2
        off_y = (out_h - in_h) // 2
        if fmt.chroma_w:
            off_x -= off_x % 2
        if fmt.chroma_h:
            off_y -= off_y % 2
        plan = _Plan(True, out_w, out_h, in_w, in_h, off_x, off_y)
        log.debug(
            "Target %s: %dx%d -> %dx%d inside %dx%d at (%d, %d)",
            target.name, src_w, src_h, in_w, in_h, out_w, out_h, off_x, off_y,
        )
        return plan

    def _fit(
        self,
        planes: Sequence[torch.Tensor],
        fmt: PixFmt,
        plan: _Plan,
        pads: Sequence[float],
        limits: Optional[Sequence[Tuple[float, float]]],
        on_chroma_grid: Optional[Sequence[bool]] = None,
    ) -> List[torch.Tensor]:
        """Scale finished float planes into the target frame and pad the rest with constants.

        Each plane is scaled on its own grid (chroma at chroma resolution), so no extra chroma
        resampling is introduced; ``on_chroma_grid`` says which planes live on the chroma grid
        (default: the chroma planes of *fmt*).  Bicubic interpolation overshoots at edges, hence the
        clamp to ``limits[plane] = (low, high)`` - the code range for pictures, black..white for a
        matte; ``limits=None`` leaves the clamping to the caller.  A constant region stays that
        constant: the filter weights sum to one up to float rounding, which the final integer
        rounding removes.  The returned planes may be the given ones (never modify them in place).
        """

        if on_chroma_grid is None:
            on_chroma_grid = [fmt.is_chroma_plane(index) for index in range(len(planes))]
        out: List[torch.Tensor] = []
        for index, plane in enumerate(planes):
            chroma = bool(on_chroma_grid[index])
            sub_w, sub_h = (fmt.chroma_w, fmt.chroma_h) if chroma else (0, 0)
            full_h, full_w = fmt.plane_size(1 if chroma else 0, plan.out_w, plan.out_h)
            in_h, in_w = plan.in_h >> sub_h, plan.in_w >> sub_w
            scaled = resize(plane, (in_h, in_w), mode="bicubic", antialias=True)
            if limits is not None:
                # Out of place: ``resize`` returns its input when the size already matches, and
                # that input may be a source plane.
                scaled = scaled.clamp(limits[index][0], limits[index][1])
            if (in_h, in_w) == (full_h, full_w):
                out.append(scaled)
                continue
            canvas = torch.full(
                (plane.shape[0], 1, full_h, full_w), float(pads[index]), dtype=torch.float32, device=plane.device
            )
            top, left = plan.off_y >> sub_h, plan.off_x >> sub_w
            canvas[:, :, top:top + in_h, left:left + in_w] = scaled
            out.append(canvas)
        return out

    def _fit_straight(
        self, planes: Sequence[torch.Tensor], fmt: PixFmt, plan: _Plan, pads: Sequence[float]
    ) -> List[torch.Tensor]:
        """Scale a straight-alpha picture: resample premultiplied, then divide the alpha out again.

        ``planes`` are the *premultiplied* colour planes (colour over black) followed by the alpha
        plane, at source size.  Resampling straight colour would average the subject's colour with
        whatever sits under the transparent samples next to it and leave a dark fringe; resampling
        ``a*F`` and ``a`` with the same filter and dividing afterwards does not.

        Per plane ``S = K + (P - K) * max / A`` with the black level ``K`` and the alpha ``A`` *as
        that plane saw it*: subsampled chroma planes are divided by an alpha that was box-averaged
        to the chroma grid before scaling, i.e. filtered exactly like the chroma itself, so the
        quotient is exact for a uniformly coloured subject.  The division uses the unclamped filter
        outputs (colour and alpha ring alike at an edge and cancel).  The gain is capped at 255 so
        samples that are all but invisible cannot amplify noise into wild colours, and samples
        whose alpha rounds to zero are plain black.
        """

        count = fmt.color_planes
        peak = float(fmt.max_value)
        subsampled = fmt.family == "yuv" and fmt.is_subsampled
        work = list(planes)
        work_pads = list(pads)
        grid = [fmt.is_chroma_plane(index) for index in range(len(work))]
        if subsampled:
            work.append(downsample_chroma(planes[count], fmt.chroma_w, fmt.chroma_h))
            work_pads.append(0.0)
            grid.append(True)
        scaled = self._fit(work, fmt, plan, work_pads, None, grid)
        alpha = scaled[count]
        # "Transparent" is decided on the alpha that is actually stored: a sample none of whose
        # pixels will be visible (alpha rounds to 0) is black, as in the unscaled case.
        clear = alpha < 0.5
        chroma_alpha, chroma_clear = alpha, clear
        if subsampled:
            chroma_alpha = scaled[count + 1]
            kernel = (1 << fmt.chroma_h, 1 << fmt.chroma_w)
            chroma_clear = F.max_pool2d(alpha, kernel_size=kernel, stride=kernel, ceil_mode=True) < 0.5

        floor = peak / 255.0
        out: List[torch.Tensor] = []
        for index in range(count):
            on_chroma = fmt.is_chroma_plane(index)
            plane_alpha = chroma_alpha if on_chroma else alpha
            level = float(pads[index])
            gain = peak / plane_alpha.clamp(min=floor)
            straight = (scaled[index] - level).mul_(gain).add_(level).clamp_(0.0, peak)
            out.append(torch.where(chroma_clear if on_chroma else clear, level, straight))
        out.append(alpha.clamp(0.0, peak))
        return out


# --------------------------------------------------------------------------- previews


def _checkerboard(height: int, width: int, device: torch.device) -> torch.Tensor:
    """``[1, 3, height, width]`` grey checkerboard, the conventional picture of "transparent"."""

    cell = max(4, min(height, width) // 12)
    rows = torch.arange(height, device=device).div(cell, rounding_mode="floor").view(height, 1)
    cols = torch.arange(width, device=device).div(cell, rounding_mode="floor").view(1, width)
    odd = ((rows + cols) % 2).to(torch.float32)
    board = odd * (_CHECKER_GREYS[1] - _CHECKER_GREYS[0]) + _CHECKER_GREYS[0]
    return board.view(1, 1, height, width).expand(1, 3, height, width)


@torch.no_grad()
def preview_atlas(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    background: Optional[Tuple[int, int, int]],
    tile_size: Tuple[int, int],
) -> torch.Tensor:
    """2 x 2 monitoring picture, ``uint8 [B, 2*th, 2*tw, 3]`` RGB, for ``tile_size = (tw, th)``.

    ``rgb`` (``[B, 3, H, W]``) and ``alpha`` (``[B, 1, H, W]``) are in **display** orientation.

    ====================  ==========================================================================
    top-left              the original frame
    top-right             "segmented": the subject at full brightness, everything else desaturated
                          and dimmed to about 35 percent, with a thin bright outline along the
                          alpha = 0.5 contour
    bottom-left           the matte as grey
    bottom-right          the composite over *background* (``a*rgb + (1-a)*bg``); a grey
                          checkerboard stands for transparency when *background* is None
    ====================  ==========================================================================

    The frame is area-averaged down to the tile size first and the four tiles are built at that
    size: the preview is a monitoring aid, so it deliberately skips foreground estimation and the
    plane-exact compositor.  The tile is filled completely; pick a tile size with the frame's
    aspect ratio to avoid distortion.
    """

    if rgb.dim() != 4 or rgb.shape[1] != 3:
        raise ValueError(f"preview_atlas: rgb must be [B, 3, H, W], got shape {tuple(rgb.shape)}")
    if alpha.dim() != 4 or alpha.shape[1] != 1 or alpha.shape[0] != rgb.shape[0] or alpha.shape[-2:] != rgb.shape[-2:]:
        raise ValueError(
            f"preview_atlas: alpha must be [B, 1, H, W] matching rgb {tuple(rgb.shape)}, got {tuple(alpha.shape)}"
        )
    try:
        tile_w, tile_h = int(tile_size[0]), int(tile_size[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise ConfigError(f"Preview tile size must be (width, height), got {tile_size!r}.") from exc
    if tile_w <= 0 or tile_h <= 0:
        raise ConfigError(f"Preview tile size must be positive, got {tile_size!r}.")
    if background is not None and (len(background) != 3 or not all(0 <= int(v) <= 255 for v in background)):
        raise ConfigError(f"Preview background must be three values in 0..255, got {background!r}.")

    device = rgb.device
    size = (tile_h, tile_w)
    image = resize(rgb.to(torch.float32), size, mode="bilinear", antialias=True).clamp(0.0, 1.0)
    matte = torch.nan_to_num(alpha.to(device=device, dtype=torch.float32), nan=0.0)
    matte = resize(matte, size, mode="bilinear", antialias=True).clamp(0.0, 1.0)

    # top-right: subject in colour, the rest as dimmed grey, outlined at the 0.5 contour
    dimmed = bt709_luma(image) * _PREVIEW_DIM
    segmented = (image - dimmed).mul_(matte).add_(dimmed)
    inside = (matte >= 0.5).to(torch.float32)
    # One-pixel ring just outside the alpha >= 0.5 region: thin, and it never covers the subject
    # itself, so even hair-thin structures stay visible between their two outline pixels.
    outline = F.max_pool2d(inside, kernel_size=3, stride=1, padding=1) > inside
    outline_colour = torch.tensor(
        [v / 255.0 for v in _PREVIEW_OUTLINE], dtype=torch.float32, device=device
    ).view(1, 3, 1, 1)
    segmented = torch.where(outline, outline_colour, segmented)

    # bottom-right: simple composite over the background colour or a checkerboard
    if background is None:
        backdrop = _checkerboard(tile_h, tile_w, device)
    else:
        backdrop = torch.tensor(
            [int(v) / 255.0 for v in background], dtype=torch.float32, device=device
        ).view(1, 3, 1, 1)
    composite = (image * matte).add_((1.0 - matte) * backdrop)

    top = torch.cat([image, segmented], dim=3)
    bottom = torch.cat([matte.expand(-1, 3, -1, -1), composite], dim=3)
    atlas = torch.cat([top, bottom], dim=2).mul_(255.0)
    return to_integer_planes([atlas], make_pixfmt("rgb", 8))[0].permute(0, 2, 3, 1).contiguous()


_PREVIEW_FMT = make_pixfmt("yuv", 8, 1, 1)
_PREVIEW_CONVERTER = ColorConverter(_PREVIEW_FMT, ResolvedColor(matrix="bt709", kr=0.2126, kb=0.0722, full_range=False))


@torch.no_grad()
def rgb8_to_yuv420p(rgb_u8: torch.Tensor) -> List[torch.Tensor]:
    """``uint8 [B, H, W, 3]`` (even H and W) -> BT.709 limited-range ``yuv420p`` integer planes.

    Returns ``[B, 1, H, W]``, ``[B, 1, H/2, W/2]``, ``[B, 1, H/2, W/2]`` (``uint8``) for the preview
    encoder.  The preview is always BT.709 / limited range, independent of the source's colour
    description, because it is a display picture, not a deliverable.
    """

    if rgb_u8.dim() != 4 or rgb_u8.shape[-1] != 3:
        raise ValueError(f"rgb8_to_yuv420p: expected [B, H, W, 3], got shape {tuple(rgb_u8.shape)}")
    height, width = int(rgb_u8.shape[1]), int(rgb_u8.shape[2])
    if height % 2 or width % 2:
        raise ConfigError(
            f"yuv420p needs even dimensions, got {width}x{height}.",
            hint="Use an even preview tile size.",
        )
    rgb = rgb_u8.permute(0, 3, 1, 2).contiguous().to(torch.float32).div(255.0)
    return to_integer_planes(_PREVIEW_CONVERTER.rgb_to_planes(rgb), _PREVIEW_FMT)
