// The processing configuration as the user edits it.
//
// ConfigModel keeps a *baseline* (the server's defaults from /api/options, or the stored config of
// a job that is being re-rendered) and a sparse set of *overrides*: only values the user changed.
// `payload()` returns exactly those overrides as the nested "config" object for the API; everything
// else is left to the server, and "automatic" stays null.
//
// The descriptors further down drive the settings form. Limits mirror the validation rules of
// app/core/config.py; the server stays the authority and its validation errors are shown as is.

export function deepEqual(a, b) {
  if (a === b) return true;
  if (typeof a !== typeof b || a === null || b === null || typeof a !== 'object') return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const keysA = Object.keys(a);
  const keysB = Object.keys(b);
  if (keysA.length !== keysB.length) return false;
  return keysA.every((key) => deepEqual(a[key], b[key]));
}

function readPath(object, path) {
  let node = object;
  for (const part of path.split('.')) {
    if (node === null || node === undefined || typeof node !== 'object') return undefined;
    node = node[part];
  }
  return node;
}

function writePath(object, path, value) {
  const parts = path.split('.');
  let node = object;
  for (let i = 0; i < parts.length - 1; i += 1) {
    if (typeof node[parts[i]] !== 'object' || node[parts[i]] === null) node[parts[i]] = {};
    node = node[parts[i]];
  }
  node[parts[parts.length - 1]] = value;
}

export class ConfigModel {
  constructor(baseline = null, overrides = {}) {
    this.baseline = baseline;
    this.overrides = { ...overrides };
    this.listeners = new Set();
  }

  get ready() {
    return this.baseline !== null;
  }

  setBaseline(baseline) {
    this.baseline = baseline;
    // drop overrides that no longer differ from (or no longer exist in) the baseline
    for (const path of Object.keys(this.overrides)) {
      const base = readPath(baseline, path);
      if (base === undefined || deepEqual(base, this.overrides[path])) delete this.overrides[path];
    }
    this.emit();
  }

  base(path) {
    const value = this.baseline ? readPath(this.baseline, path) : undefined;
    return value === undefined ? null : value;
  }

  get(path) {
    return Object.prototype.hasOwnProperty.call(this.overrides, path) ? this.overrides[path] : this.base(path);
  }

  isChanged(path) {
    return Object.prototype.hasOwnProperty.call(this.overrides, path);
  }

  changedPaths() {
    return Object.keys(this.overrides).sort();
  }

  /** Set several values at once; listeners are notified once. */
  update(values) {
    let changed = false;
    for (const [path, raw] of Object.entries(values)) {
      const value = raw === undefined ? null : raw;
      const had = this.isChanged(path);
      if (deepEqual(value, this.base(path))) {
        if (had) {
          delete this.overrides[path];
          changed = true;
        }
      } else if (!had || !deepEqual(this.overrides[path], value)) {
        this.overrides[path] = value;
        changed = true;
      }
    }
    if (changed) this.emit();
  }

  set(path, value) {
    this.update({ [path]: value });
  }

  reset(path) {
    if (this.isChanged(path)) {
      delete this.overrides[path];
      this.emit();
    }
  }

  resetAll(prefix = '') {
    const before = Object.keys(this.overrides).length;
    for (const path of Object.keys(this.overrides)) {
      if (!prefix || path === prefix || path.startsWith(`${prefix}.`)) delete this.overrides[path];
    }
    if (Object.keys(this.overrides).length !== before) this.emit();
  }

  /** The sparse nested object that is sent to the API. */
  payload() {
    const out = {};
    for (const path of this.changedPaths()) writePath(out, path, this.overrides[path]);
    return out;
  }

  /** Baseline with the overrides applied (full nested object). */
  effective() {
    const out = this.baseline ? JSON.parse(JSON.stringify(this.baseline)) : {};
    for (const path of this.changedPaths()) writePath(out, path, this.overrides[path]);
    return out;
  }

  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  emit() {
    for (const listener of [...this.listeners]) {
      try {
        listener(this);
      } catch (error) {
        console.error(error);
      }
    }
  }
}

// ------------------------------------------------------------------ catalogue helpers

export function codecInfo(options, codecId) {
  return options && codecId ? options.codecs.find((c) => c.id === codecId) || null : null;
}

export function qualityInfo(options, qualityId) {
  return options ? options.quality_presets.find((q) => q.id === qualityId) || null : null;
}

/** Name of the refinement preset that is in effect (explicit, or the one of the quality preset). */
export function effectiveRefinementPreset(model, options) {
  const explicit = model.get('refinement.preset');
  if (explicit) return explicit;
  const quality = qualityInfo(options, model.get('quality'));
  return quality ? quality.refinement_preset : null;
}

/** Value a refinement field has when the user leaves it on "preset" (null). */
export function presetRefinementValue(model, options, key) {
  const name = effectiveRefinementPreset(model, options);
  const preset = name && options ? options.refinement_presets[name] : null;
  return preset && Object.prototype.hasOwnProperty.call(preset, key) ? preset[key] : null;
}

/** Containers that are valid for the current codec / background combination. */
export function containerChoices(model, options) {
  const codec = codecInfo(options, model.get('encoding.codec'));
  if (codec) return codec.containers.slice();
  if (model.get('background') === 'transparent') {
    // automatic codec with transparency: ProRes 4444 (.mov), VP9 (.webm) or FFV1 (.mkv)
    const alpha = new Set();
    for (const c of options.codecs) if (c.supports_alpha) c.containers.forEach((name) => alpha.add(name));
    return options.containers.filter((name) => alpha.has(name));
  }
  return options.containers.slice();
}

/**
 * Keep dependent settings consistent after `path` changed. Returns short sentences describing
 * what was adjusted (shown next to the control), or an empty list.
 */
export function reconcile(model, options, path) {
  const notes = [];
  if (!options) return notes;
  const patch = {};
  const codecId = model.get('encoding.codec');
  const codec = codecInfo(options, codecId);

  if (path === 'model') {
    if (model.get('variant') !== null) {
      patch.variant = null;
      notes.push('The variant was reset to automatic for the new model.');
    }
  }

  if (path === 'background') {
    const background = model.get('background');
    if (background === 'custom') {
      if (!model.get('background_color')) patch.background_color = '#00FF00';
    } else if (model.get('background_color') !== null) {
      patch.background_color = null;
    }
    if (background === 'transparent' && codec && !codec.supports_alpha) {
      patch['encoding.codec'] = null;
      patch['encoding.container'] = null;
      Object.assign(patch, codecSpecificReset());
      notes.push(`${codec.label} has no alpha channel: the codec was set to automatic (ProRes 4444, VP9 or FFV1).`);
    } else if (background === 'transparent' && !codec) {
      const allowed = containerChoices(model, options);
      const container = model.get('encoding.container');
      if (container && !allowed.includes(container)) {
        patch['encoding.container'] = null;
        notes.push(`Transparent output cannot be stored in .${container}: the container was set to automatic.`);
      }
    }
  }

  if (path === 'encoding.codec') {
    Object.assign(patch, codecSpecificReset());
    const container = model.get('encoding.container');
    if (container && !containerChoices(model, options).includes(container)) {
      patch['encoding.container'] = null;
      notes.push(`.${container} cannot hold this codec: the container was set to automatic.`);
    }
  }

  if (Object.keys(patch).length) model.update(patch);
  // the codec that will really be used may have changed (also with the codec on automatic)
  if (path === 'background' || path === 'encoding.codec' || path === 'encoding.container') {
    notes.push(...dropUnusableEncoderSettings(model));
  }
  return notes;
}

/** Set encoder settings back to automatic when they cannot be used with the codec that is planned now. */
function dropUnusableEncoderSettings(model) {
  const notes = [];
  const patch = {};
  const codecId = plannedCodecId(model);
  const range = QUALITY_RANGE[codecId] || null;
  const crf = model.get('encoding.crf');
  if (crf !== null && (!range || crf < range[0] || crf > range[1])) {
    patch['encoding.crf'] = null;
    notes.push('The constant-quality value does not apply to this codec: it was set to automatic.');
  }
  if (!range && (model.get('encoding.bitrate') !== null || model.get('encoding.max_bitrate') !== null)) {
    patch['encoding.bitrate'] = null;
    patch['encoding.max_bitrate'] = null;
    notes.push('This codec has no rate control: the bitrate was set to automatic.');
  }
  const preset = model.get('encoding.preset');
  if (preset !== null && !(ENCODER_PRESETS[codecId] || []).includes(preset)) {
    patch['encoding.preset'] = null;
    notes.push('The encoder preset does not exist for this codec: it was set to automatic.');
  }
  const pixFmt = model.get('encoding.pix_fmt');
  const usable = pixelFormatChoices(codecId, model.get('background') === 'transparent');
  if (pixFmt !== null && !usable.some((choice) => choice.value === pixFmt)) {
    patch['encoding.pix_fmt'] = null;
    notes.push('The pixel format cannot be used with this output: it was set to automatic.');
  }
  if (Object.keys(patch).length) model.update(patch);
  return notes;
}

function codecSpecificReset() {
  return {
    'encoding.crf': null,
    'encoding.bitrate': null,
    'encoding.max_bitrate': null,
    'encoding.preset': null,
    'encoding.pix_fmt': null,
  };
}

// ------------------------------------------------------------------ descriptors for the form

/** Encoder speed presets per codec id (FFmpeg names; see EncodingConfig.preset). */
export const ENCODER_PRESETS = {
  h264: ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
  h265: ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
  h264_nvenc: ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'],
  hevc_nvenc: ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'],
  vp9: ['0', '1', '2', '3', '4', '5', '6', '7', '8'],
};

/** The choices of the encoder preset select for a codec: [{value, label}] (empty = the codec has none). */
export function encoderPresetChoices(codecId) {
  const names = ENCODER_PRESETS[codecId] || [];
  return names.map((name, index) => {
    let label = name;
    if (codecId === 'vp9') label = `Speed ${name}${index === 0 ? ' (slowest, best compression)' : index === names.length - 1 ? ' (fastest)' : ''}`;
    else if (/^p\d$/.test(name)) label = `${name}${index === 0 ? ' (fastest)' : index === names.length - 1 ? ' (best quality)' : ''}`;
    return { value: name, label };
  });
}

/** Range of the constant-quality value per codec (CRF for x264 / x265 / VP9, CQ for NVENC). */
export const QUALITY_RANGE = {
  h264: [0, 51],
  h265: [0, 51],
  h264_nvenc: [1, 51],
  hevc_nvenc: [1, 51],
  vp9: [0, 63],
};

const YUV_8_AND_10 = ['yuv420p', 'yuv422p', 'yuv444p', 'yuv420p10le', 'yuv422p10le', 'yuv444p10le'];

/**
 * Pixel formats offered besides "automatic", per codec. Only layouts every build of these
 * encoders accepts are listed; codecs whose layout is fixed by their profile (ProRes) and
 * transparent VP9 (8-bit 4:2:0 with alpha is the only layout browsers play) offer none.
 */
const PIXEL_FORMATS = {
  h264: { opaque: YUV_8_AND_10, alpha: [] },
  h265: { opaque: YUV_8_AND_10, alpha: [] },
  h264_nvenc: { opaque: ['yuv420p', 'yuv444p'], alpha: [] },
  hevc_nvenc: { opaque: ['yuv420p', 'yuv444p', 'yuv420p10le'], alpha: [] },
  prores_422_hq: { opaque: [], alpha: [] },
  prores_4444: { opaque: [], alpha: [] },
  ffv1: { opaque: YUV_8_AND_10, alpha: ['yuva420p', 'yuva422p', 'yuva444p', 'yuva444p10le'] },
  vp9: { opaque: YUV_8_AND_10, alpha: [] },
};

/** "yuv422p10le" -> "4:2:2, 10-bit (yuv422p10le)". */
export function pixelFormatLabel(name) {
  const match = String(name).match(/^yuva?(4\d\d)p(\d+)?(?:le|be)?$/);
  if (!match) return String(name);
  const chroma = `${match[1][0]}:${match[1][1]}:${match[1][2]}`;
  const alpha = String(name).startsWith('yuva') ? ' + alpha' : '';
  return `${chroma}${alpha}, ${match[2] || '8'}-bit (${name})`;
}

export function pixelFormatChoices(codecId, transparent) {
  const entry = PIXEL_FORMATS[codecId];
  const names = entry ? (transparent ? entry.alpha : entry.opaque) : [];
  return names.map((name) => ({ value: name, label: pixelFormatLabel(name) }));
}

/**
 * The codec the server will choose for the main output, as far as the settings decide it. With
 * everything on automatic the source decides between H.264 (8-bit) and H.265 (deeper); both share
 * the same controls, so 'h264' stands for that case.
 */
export function plannedCodecId(model) {
  const explicit = model.get('encoding.codec');
  if (explicit) return explicit;
  const container = model.get('encoding.container');
  if (model.get('background') === 'transparent') return { mov: 'prores_4444', webm: 'vp9', mkv: 'ffv1' }[container || 'mov'] || null;
  return container === 'webm' ? 'vp9' : 'h264';
}

/** The container the output gets when it is left on automatic. */
export function automaticContainer(model, options) {
  const codec = codecInfo(options, model.get('encoding.codec'));
  if (codec) return codec.containers[0];
  return model.get('background') === 'transparent' ? 'mov' : 'mp4';
}

/** The API accepts bit rates such as 8000k, 20M or 5000000. */
export const BITRATE_PATTERN = /^\d+(?:\.\d+)?[kKmM]?$/;

export const CODEC_GROUPS = [
  { id: 'delivery', label: 'Delivery' },
  { id: 'intermediate', label: 'Intermediate' },
  { id: 'alpha', label: 'Alpha channel' },
  { id: 'lossless', label: 'Lossless' },
];

export const RESOLUTION_PRESETS = ['2160p', '1440p', '1080p', '720p'];

export const AUDIO_CODECS = [
  { value: 'auto', label: 'Automatic (copy when the container allows it)' },
  { value: 'copy', label: 'Copy the original stream' },
  { value: 'aac', label: 'AAC' },
  { value: 'opus', label: 'Opus' },
  { value: 'flac', label: 'FLAC (lossless)' },
  { value: 'pcm', label: 'PCM (uncompressed)' },
];

/**
 * Refinement controls, grouped. `scale` converts the API value for display (areas are stored as
 * a fraction of the frame and shown in percent); `soft` is the range of the slider while the
 * number box accepts the full `min`..`max` range of the API.
 */
export const REFINEMENT_GROUPS = [
  {
    title: 'Clean-up',
    fields: [
      { key: 'remove_small_artifacts', type: 'bool', label: 'Remove small artefacts', hint: 'Deletes isolated specks that are not connected to the subject.' },
      { key: 'min_artifact_area', type: 'number', label: 'Largest speck removed', unit: '% of frame', scale: 100, min: 0, max: 0.5, step: 0.0001, soft: [0, 0.01], digits: 2, hint: 'Specks smaller than this share of the frame area are removed.' },
      { key: 'fill_holes', type: 'bool', label: 'Fill holes', hint: 'Closes small holes inside the subject.' },
      { key: 'max_hole_area', type: 'number', label: 'Largest hole filled', unit: '% of frame', scale: 100, min: 0, max: 0.5, step: 0.0001, soft: [0, 0.01], digits: 2, hint: 'Holes smaller than this share of the frame area are filled.' },
    ],
  },
  {
    title: 'Edge shape',
    fields: [
      { key: 'erode', type: 'number', label: 'Erode (shrink)', unit: 'px', min: 0, max: 64, step: 0.5, soft: [0, 12], digits: 1, hint: 'Pulls the matte edge inwards.' },
      { key: 'dilate', type: 'number', label: 'Dilate (grow)', unit: 'px', min: 0, max: 64, step: 0.5, soft: [0, 12], digits: 1, hint: 'Pushes the matte edge outwards.' },
      { key: 'edge_feather', type: 'number', label: 'Edge feather', unit: 'px', min: 0, max: 64, step: 0.5, soft: [0, 12], digits: 1, hint: 'Softens the transition at the matte edge.' },
      { key: 'gaussian_smoothing', type: 'number', label: 'Gaussian smoothing', unit: 'sigma', min: 0, max: 32, step: 0.25, soft: [0, 8], digits: 2, hint: 'Blurs the whole matte; removes stair-stepping.' },
    ],
  },
  {
    title: 'Temporal stability',
    fields: [
      { key: 'temporal_smoothing', type: 'number', label: 'Temporal smoothing', min: 0, max: 0.95, step: 0.05, digits: 2, hint: '0 switches it off; higher values suppress flicker more strongly.' },
      { key: 'motion_aware', type: 'bool', label: 'Motion aware', hint: 'Smooths only where the picture is static, so moving edges leave no trails.' },
      { key: 'motion_compensation', type: 'bool', label: 'Motion compensation', hint: 'Warps the previous matte along optical flow before blending (slower).' },
    ],
  },
  {
    title: 'Detail and levels',
    fields: [
      { key: 'preserve_hair', type: 'bool', label: 'Preserve hair detail', hint: 'Keeps fine semi-transparent strands through the clean-up steps.' },
      { key: 'preserve_motion_blur', type: 'bool', label: 'Preserve motion blur', hint: 'Keeps soft, blurred edges of fast movement.' },
      { key: 'black_point', type: 'number', label: 'Black point', min: 0, max: 0.45, step: 0.005, soft: [0, 0.2], digits: 3, hint: 'Alpha at or below this value becomes fully transparent.' },
      { key: 'white_point', type: 'number', label: 'White point', min: 0.55, max: 1, step: 0.005, soft: [0.8, 1], digits: 3, hint: 'Alpha at or above this value becomes fully opaque.' },
      { key: 'decontaminate', type: 'number', label: 'Edge colour decontamination', min: 0, max: 1, step: 0.05, digits: 2, hint: '0 keeps the source pixels at edges exactly; 1 removes the old background colour from semi-transparent edges.' },
    ],
  },
  {
    title: 'Mask to alpha',
    fields: [
      {
        key: 'mask_to_alpha',
        type: 'enum',
        label: 'Mask to alpha conversion',
        options: [
          { value: 'auto', label: 'Automatic (guided filter for mask models)' },
          { value: 'guided_filter', label: 'Guided filter' },
          { value: 'none', label: 'None' },
        ],
        hint: 'How a hard segmentation mask is turned into a soft matte.',
      },
      { key: 'guided_radius', type: 'number', label: 'Guided filter radius', unit: 'px', min: 1, max: 64, step: 0.5, soft: [1, 24], digits: 1, hint: 'Width of the band in which the matte follows image edges.' },
      { key: 'guided_eps', type: 'number', label: 'Guided filter epsilon', min: 0.000001, max: 1, step: 0.0001, digits: 4, noSlider: true, hint: 'Smaller values follow image edges more closely.' },
      { key: 'resolution_scaling', type: 'bool', label: 'Scale pixel values with the resolution', hint: 'Pixel quantities are given for a 1080-line frame and scaled to the real frame size.' },
    ],
  },
];

export const REFINEMENT_KEYS = REFINEMENT_GROUPS.flatMap((group) => group.fields.map((field) => field.key));

/**
 * Refinement controls that only act while another one is switched on: key -> test on the
 * effective values. The form greys them out otherwise (their values are still sent when changed).
 */
export const REFINEMENT_NEEDS = {
  min_artifact_area: (value) => value('remove_small_artifacts') === true,
  max_hole_area: (value) => value('fill_holes') === true,
  guided_radius: (value) => value('mask_to_alpha') !== 'none',
  guided_eps: (value) => value('mask_to_alpha') !== 'none',
  motion_aware: (value) => Number(value('temporal_smoothing')) > 0,
  motion_compensation: (value) => Number(value('temporal_smoothing')) > 0,
};

export const REFINEMENT_PRESET_LABELS = { off: 'Off', light: 'Light', balanced: 'Balanced', strong: 'Strong' };

export const EXPORTS = [
  { key: 'alpha_matte', label: 'Alpha matte video', hint: 'Greyscale matte (white = subject).' },
  { key: 'png_sequence', label: 'PNG sequence', hint: 'RGBA image sequence in a folder (offered as .zip).' },
  { key: 'prores_4444', label: 'Transparent ProRes 4444', hint: '.mov with an alpha channel.' },
  { key: 'green_screen', label: 'Green-screen video', hint: 'Additional green version when the main output uses another background.' },
  { key: 'audio', label: 'Original audio', hint: 'Audio track(s), stream-copied without re-encoding.' },
  { key: 'matte_preview', label: 'Matte preview', hint: 'Low resolution 2 x 2 video: original, subject, matte, composite.' },
];

/** Settings that decide the raw matte: changing one of them makes a re-render run the model again. */
export const MATTE_FINGERPRINT = [
  { path: 'model', label: 'model' },
  { path: 'variant', label: 'variant' },
  { path: 'quality', label: 'quality preset' },
  { path: 'precision', label: 'precision' },
  { path: 'max_frames', label: 'frame limit' },
];

// ------------------------------------------------------------------ naming settings for summaries

const SETTING_LABELS = {
  model: 'Model',
  variant: 'Model variant',
  quality: 'Quality preset',
  device: 'Device',
  precision: 'Precision',
  batch_size: 'Batch size',
  background: 'Background',
  background_color: 'Background colour',
  max_frames: 'Frame limit',
  strict_verification: 'Strict verification',
  commercial_use: 'Commercial use',
  download_models: 'Download missing weights',
  allow_cpu_fallback: 'Fall back to the CPU',
  tolerate_decode_errors: 'Skip undecodable frames',
  'refinement.preset': 'Refinement preset',
  'encoding.codec': 'Codec',
  'encoding.container': 'Container',
  'encoding.crf': 'Constant quality',
  'encoding.bitrate': 'Target bitrate',
  'encoding.max_bitrate': 'Maximum bitrate',
  'encoding.preset': 'Encoder preset',
  'encoding.pix_fmt': 'Pixel format',
  'encoding.resolution': 'Resolution',
  'encoding.preserve_audio': 'Keep the original audio',
  'encoding.audio_codec': 'Audio codec',
  'encoding.audio_bitrate': 'Audio bitrate',
  'exports.png_bit_depth': 'PNG bit depth',
  'exports.alpha_mode': 'Alpha mode',
};
for (const group of REFINEMENT_GROUPS) for (const item of group.fields) SETTING_LABELS[`refinement.${item.key}`] = `Refinement: ${item.label.toLowerCase()}`;
for (const item of EXPORTS) SETTING_LABELS[`exports.${item.key}`] = `Export: ${item.label.toLowerCase()}`;

/** A readable name for a settings path ("encoding.crf" -> "Constant quality"). */
export function settingLabel(path) {
  return SETTING_LABELS[path] || path;
}

/** A settings value in words: null is "automatic", switches are "on" / "off". */
export function describeSettingValue(path, value, options = null) {
  if (value === null || value === undefined) return 'automatic';
  if (typeof value === 'boolean') return value ? 'on' : 'off';
  if (path === 'encoding.codec') {
    const codec = codecInfo(options, value);
    return codec ? codec.label : String(value);
  }
  if (path === 'encoding.container') return `.${value}`;
  if (path === 'encoding.resolution') return value === 'source' ? 'original' : String(value);
  if (path === 'refinement.preset') return REFINEMENT_PRESET_LABELS[value] || String(value);
  if (path === 'quality') {
    const quality = qualityInfo(options, value);
    return quality ? quality.label : String(value);
  }
  if (typeof value === 'number') return String(parseFloat(value.toPrecision(6)));
  if (Array.isArray(value) || typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

// ------------------------------------------------------------------ subject prompt

export function defaultSubject(options) {
  const base = (options && options.defaults && options.defaults.subject) || {};
  return {
    mode: 'auto',
    frame_index: 0,
    points: [],
    box: null,
    text: '',
    mask_path: '',
    isolate: base.isolate !== false,
    track_backward: base.track_backward !== false,
  };
}

/** A human summary of a subject selection: "Automatic", "3 clicks on frame 12", ... */
export function describeSubject(subject) {
  if (!subject) return 'Automatic';
  const frame = subject.frame_index ? ` on frame ${subject.frame_index}` : '';
  switch (subject.mode) {
    case 'points': {
      const positive = subject.points.filter((p) => p.label === 1).length;
      const negative = subject.points.length - positive;
      const parts = [`${positive} subject click${positive === 1 ? '' : 's'}`];
      if (negative) parts.push(`${negative} exclusion${negative === 1 ? '' : 's'}`);
      return parts.join(', ') + frame;
    }
    case 'box':
      return subject.box ? `Box${frame}` : 'Box (not drawn yet)';
    case 'text':
      return subject.text.trim() ? `Text: “${subject.text.trim()}”${frame}` : 'Text (empty)';
    case 'mask':
      return subject.mask_path.trim() ? `Mask image${frame}` : 'Mask image (no path)';
    default:
      return subject.frame_index ? `Automatic, from frame ${subject.frame_index}` : 'Automatic';
  }
}

/** Why a selection cannot be sent yet (null when it is complete). */
export function subjectProblem(subject) {
  if (!subject) return null;
  if (subject.mode === 'points' && !subject.points.some((p) => p.label === 1)) return 'Click the subject at least once.';
  if (subject.mode === 'box' && !subject.box) return 'Drag a box around the subject.';
  if (subject.mode === 'text' && !subject.text.trim()) return 'Describe the subject.';
  if (subject.mode === 'mask' && !subject.mask_path.trim()) return 'Enter the path of the mask image.';
  return null;
}

/** The SubjectPrompt object for the API: the mode, its data, and only the switches that differ from the defaults. */
export function subjectPayload(subject, options) {
  const base = defaultSubject(options);
  const out = { mode: subject.mode, frame_index: subject.frame_index };
  if (subject.mode === 'points') out.points = subject.points.map((p) => ({ x: round2(p.x), y: round2(p.y), label: p.label }));
  if (subject.mode === 'box' && subject.box) out.box = subject.box.map(round2);
  if (subject.mode === 'text') out.text = subject.text.trim();
  if (subject.mode === 'mask') out.mask_path = subject.mask_path.trim();
  if (subject.isolate !== base.isolate) out.isolate = subject.isolate;
  if (subject.track_backward !== base.track_backward) out.track_backward = subject.track_backward;
  return out;
}

/** True when the prompt equals the server default (nothing has to be sent). */
export function isDefaultSubject(subject, options) {
  if (!subject) return true;
  const base = defaultSubject(options);
  return subject.mode === 'auto' && subject.frame_index === 0 && subject.isolate === base.isolate && subject.track_backward === base.track_backward;
}

function round2(value) {
  return Math.round(value * 100) / 100;
}

// ------------------------------------------------------------------ validation errors

const KEYWORD_FIELDS = [
  [/background_color|background 'custom'/i, 'background_color'],
  [/cannot be stored in|container/i, 'encoding.container'],
  [/alpha channel|transparent/i, 'encoding.codec'],
  [/device must be/i, 'device'],
  [/bitrate/i, 'encoding.bitrate'],
  [/resolution/i, 'encoding.resolution'],
];

/**
 * Split a validation message of the API ("Invalid settings: encoding.crf: ...; background_color: ...")
 * into messages per settings path plus general messages that belong to no single control.
 */
export function parseValidationMessage(message) {
  const fields = {};
  const general = [];
  let text = String(message || '');
  const prefix = text.match(/^Invalid (settings|subject selection|options):\s*/i);
  if (prefix) text = text.slice(prefix[0].length);
  const parts = [];
  for (const piece of text.split('; ')) {
    const match = piece.match(/^([a-z_]+(?:\.[a-z_0-9]+)*):\s+(.*)$/s);
    if (match && isConfigPath(match[1])) parts.push({ path: match[1], text: match[2] });
    else if (parts.length) parts[parts.length - 1].text += `; ${piece}`;
    else parts.push({ path: '', text: piece });
  }
  for (const part of parts) {
    let path = part.path;
    if (!path) {
      const hit = KEYWORD_FIELDS.find(([pattern]) => pattern.test(part.text));
      path = hit ? hit[1] : '';
    }
    if (path) fields[path] = fields[path] ? `${fields[path]} ${part.text}` : part.text;
    else general.push(part.text);
  }
  return { fields, general };
}

const TOP_LEVEL = new Set([
  'model', 'variant', 'quality', 'device', 'precision', 'batch_size', 'background', 'background_color',
  'max_frames', 'strict_verification', 'commercial_use', 'download_models', 'allow_cpu_fallback',
  'tolerate_decode_errors', 'model_options', 'resume', 'keep_work_dir', 'overwrite',
]);
const SECTIONS = new Set(['subject', 'refinement', 'encoding', 'exports']);

function isConfigPath(path) {
  const [head] = path.split('.');
  return TOP_LEVEL.has(head) || SECTIONS.has(head);
}
