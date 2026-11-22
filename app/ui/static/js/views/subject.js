// Subject: choose who (or what) is kept. A reference frame of the active video with a scrubber,
// tools to click / drag a box / describe / point to a mask image, markers on a canvas, and a
// server-side preview of what the selection resolves to. Coordinates are sent in pixels of the
// display-oriented reference frame (the JPEG shown here may be down-scaled; it is converted).

import { h, fill, clamp, createScope, debounce, isEditableTarget } from '../dom.js';
import { icon } from '../icons.js';
import { api, toApiError } from '../api.js';
import { store, config, activeVideo, selectedVideos, subjectFor, currentModel } from '../state.js';
import { describeSubject, isDefaultSubject, subjectPayload, subjectProblem } from '../config.js';
import { formatInt, formatPercent, humanize, isNumber, plural } from '../format.js';
import { navigate, resetSubject, setActive, updateSubject } from '../actions.js';
import { badge, bullets, busyWhile, button, card, emptyState, errorNote, kv, note, segmented, skeletonCards, switchControl } from '../components/controls.js';
import { toast } from '../components/toast.js';

const TOOLS = [
  { value: 'auto', label: 'Auto', icon: 'wand', tip: 'Let the engine pick the main subject' },
  { value: 'points', label: 'Click', icon: 'pointer', tip: 'Left click = subject, right click = not the subject' },
  { value: 'box', label: 'Box', icon: 'box', tip: 'Drag a box around the subject' },
  { value: 'text', label: 'Text', icon: 'type', tip: 'Describe the subject in words' },
  { value: 'mask', label: 'Mask image', icon: 'image', tip: 'A black / white mask image of the reference frame' },
];

const TOOL_HELP = {
  auto: 'The engine chooses the main subject itself. Nothing to draw.',
  points: 'Left click on the subject. Right click (or switch to “Not the subject”) marks areas to exclude. Click a marker again to remove it.',
  box: 'Drag a box around the subject on the frame. Drawing again replaces the box.',
  text: 'Describe the subject, for example “the woman in the red jacket”.',
  mask: 'Path of a mask image on this computer: black and white (white = subject), a cut-out with transparency, or a cut-out picture on one flat background colour such as green.',
};

const MAX_SIDE = 1280;
const POINT_RADIUS = 7; // CSS pixels

function displaySize(video) {
  const v = (video && video.metadata && video.metadata.video) || {};
  return isNumber(v.display_width) && isNumber(v.display_height) ? [v.display_width, v.display_height] : null;
}

function frameCount(video) {
  const n = video && video.metadata ? video.metadata.frame_count : null;
  return isNumber(n) && n > 0 ? n : 1;
}

// ------------------------------------------------------------------ the editor of one video

function subjectEditor(scope, video) {
  const videoId = video.id;
  const total = frameCount(video);
  const dims = displaySize(video);
  const undoStack = [];
  let negative = false; // click tool: next click marks "not the subject"
  let image = null; // loaded frame (HTMLImageElement) and the index it shows
  let imageIndex = null;
  let preview = null; // {key, result, overlay(Image)}
  let previewError = null; // {key, error}
  let drag = null; // box being drawn: {x0, y0, x1, y1} in display pixels
  let loadToken = 0;

  const current = () => subjectFor(store.get(), videoId);
  const frameW = () => (dims ? dims[0] : image ? image.naturalWidth : 1);
  const frameH = () => (dims ? dims[1] : image ? image.naturalHeight : 1);

  function commit(patch, { record = true } = {}) {
    if (record) {
      undoStack.push(JSON.parse(JSON.stringify(current())));
      if (undoStack.length > 100) undoStack.shift();
    }
    updateSubject(videoId, patch);
  }

  // ---------------------------------------------------------------- elements
  const canvas = h('canvas', { class: 'subj-canvas', testid: 'subject-canvas' });
  canvas.addEventListener('contextmenu', (event) => event.preventDefault());
  const stageMsg = h('div', { class: 'subj-stage-msg', testid: 'subject-frame-status' });
  const stage = h('div', { class: 'subj-stage', testid: 'subject-stage' }, canvas, stageMsg);

  const slider = h('input', { class: 'range subj-slider', type: 'range', min: '0', max: String(Math.max(0, total - 1)), step: '1', value: '0', 'aria-label': 'Reference frame', testid: 'subject-frame-slider' });
  const frameLabel = h('span', { class: 'subj-frame num', testid: 'subject-frame-label' });
  const stepButton = (delta, iconName, tip) => button({ icon: iconName, size: 'small', variant: 'ghost', tip, ariaLabel: tip, testid: `subject-step-${delta > 0 ? 'fwd' : 'back'}${Math.abs(delta)}`, onClick: () => goToFrame(current().frame_index + delta) });
  const scrubber = h(
    'div',
    { class: 'subj-scrubber' },
    stepButton(-10, 'rewind', 'Back 10 frames'),
    stepButton(-1, 'step-back', 'Previous frame'),
    slider,
    stepButton(1, 'step-forward', 'Next frame'),
    stepButton(10, 'fast-forward', 'Forward 10 frames'),
    frameLabel,
  );

  const tools = segmented({ options: TOOLS, value: current().mode, label: 'Selection tool', testid: 'subject-tool', onChange: (mode) => commit({ mode }) });
  const polarity = segmented({
    options: [
      { value: 'pos', label: 'Subject', icon: 'plus' },
      { value: 'neg', label: 'Not the subject', icon: 'minus' },
    ],
    value: 'pos',
    label: 'What a click marks',
    testid: 'subject-polarity',
    onChange: (value) => {
      negative = value === 'neg';
    },
  });
  const toolHelp = h('p', { class: 'field-hint', testid: 'subject-tool-help' });

  const textInput = h('textarea', { class: 'input subj-text', rows: 2, maxLength: 300, placeholder: 'the woman in the red jacket', 'aria-label': 'Description of the subject', testid: 'subject-text' });
  textInput.addEventListener('input', () => commit({ text: textInput.value }, { record: false }));
  const snapshot = () => undoStack.push(JSON.parse(JSON.stringify(current())));
  textInput.addEventListener('focus', snapshot);
  const maskInput = h('input', { class: 'input mono', type: 'text', spellcheck: false, autocomplete: 'off', placeholder: 'C:\\masks\\first_frame.png', 'aria-label': 'Path of the mask image', testid: 'subject-mask-path' });
  maskInput.addEventListener('focus', () => undoStack.push(JSON.parse(JSON.stringify(current()))));
  maskInput.addEventListener('input', () => commit({ mask_path: maskInput.value }, { record: false }));
  const textField = h('div', { class: 'subj-input' }, h('label', { class: 'field-label' }, 'Description'), textInput);
  const maskField = h('div', { class: 'subj-input' }, h('label', { class: 'field-label' }, 'Mask image path'), maskInput, h('div', { class: 'field-hint' }, 'The file is read by the server on this computer; a path copied with "Copy as path" can be pasted as it is. It has to show the reference frame. For a cut-out picture, Preview selection finds the frame it shows and moves the reference frame there.'));

  const undoButton = button({ label: 'Undo', icon: 'undo', size: 'small', testid: 'subject-undo', onClick: undo, tip: 'Undo the last change (Ctrl+Z)' });
  const clearButton = button({ label: 'Clear', icon: 'x', size: 'small', testid: 'subject-clear', onClick: clearMarks, tip: 'Remove the clicks and the box' });
  const resetButton = button({ label: 'Reset to automatic', icon: 'refresh', size: 'small', variant: 'ghost', testid: 'subject-reset', onClick: () => {
    commit({});
    resetSubject(videoId);
  } });
  const previewButton = button({ label: 'Preview selection', icon: 'eye', variant: 'primary', testid: 'subject-preview', onClick: () => busyWhile(previewButton, runPreview) });
  const summary = h('span', { class: 'subj-summary', testid: 'subject-summary' });
  const problem = h('div', { class: 'field-error', hidden: true, testid: 'subject-problem' });

  const isolate = switchControl({
    label: 'Keep every subject',
    description: 'Off: only the selected (or main) subject is kept. On: everything the model mattes is kept.',
    testid: 'subject-keep-all',
    onChange: (on) => commit({ isolate: !on }),
  });
  const backward = switchControl({
    label: 'Track backwards',
    description: 'When the reference frame is not the first frame, the subject is also followed back to frame 0.',
    testid: 'subject-track-backward',
    onChange: (on) => commit({ track_backward: on }),
  });

  const resultSlot = h('div', { testid: 'subject-result' });

  // ---------------------------------------------------------------- frame loading
  const requestFrame = debounce((index) => {
    const token = ++loadToken;
    stageMsg.dataset.kind = 'loading';
    fill(stageMsg, h('span', { class: 'spinner' }), h('span', null, `Loading frame ${formatInt(index)}…`));
    const img = new Image();
    img.decoding = 'async';
    img.onload = () => {
      if (token !== loadToken || scope.disposed) return;
      image = img;
      imageIndex = index;
      stageMsg.dataset.kind = 'none';
      fill(stageMsg);
      draw();
    };
    img.onerror = async () => {
      if (token !== loadToken || scope.disposed) return;
      let message = 'The frame could not be loaded.';
      try {
        await api.frame(videoId, index, MAX_SIDE);
      } catch (error) {
        message = toApiError(error).message;
      }
      stageMsg.dataset.kind = 'error';
      fill(stageMsg, icon('alert'), h('span', null, message));
    };
    img.src = api.frameUrl(videoId, index, MAX_SIDE);
  }, 140);

  function goToFrame(index) {
    const next = clamp(Math.round(Number(index) || 0), 0, total - 1);
    const subject = current();
    if (next === subject.frame_index) {
      slider.value = String(next);
      return;
    }
    const hadMarks = subject.points.length > 0 || !!subject.box;
    commit({ frame_index: next, points: [], box: null });
    if (hadMarks) toast({ type: 'info', title: 'Markers removed', text: 'Clicks and boxes belong to the frame they were drawn on. Undo brings them back.', timeout: 3500 });
  }

  slider.addEventListener('input', () => {
    const subject = current();
    const next = Number(slider.value);
    // while dragging only the picture follows; the reference frame is set when the slider is released
    frameLabel.textContent = `Frame ${formatInt(next)} / ${formatInt(total - 1)}`;
    if (subject.points.length || subject.box) return;
    requestFrame(next);
  });
  slider.addEventListener('change', () => goToFrame(slider.value));

  // ---------------------------------------------------------------- drawing
  function previewKey() {
    return JSON.stringify([subjectPayload(current(), store.get().options), config.get('model')]);
  }

  function draw() {
    const subject = current();
    const width = image ? image.naturalWidth : dims ? Math.min(MAX_SIDE, dims[0]) : 640;
    const height = image ? image.naturalHeight : dims ? Math.round((width * dims[1]) / dims[0]) : 360;
    const rect = canvas.getBoundingClientRect();
    const dpr = rect.width ? Math.max(1, width / rect.width) : 1; // canvas pixels per CSS pixel
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    stage.style.setProperty('--ar', String(width / height));
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, width, height);
    if (!image) return;
    ctx.drawImage(image, 0, 0, width, height);
    const sx = width / frameW();
    const sy = height / frameH();

    const shown = preview && preview.key === previewKey() && imageIndex === subject.frame_index ? preview : null;
    if (shown && shown.overlay) ctx.drawImage(shown.overlay, 0, 0, width, height);
    if (shown && shown.result.box && subject.mode !== 'box') {
      const [x1, y1, x2, y2] = shown.result.box;
      ctx.setLineDash([6 * dpr, 4 * dpr]);
      ctx.lineWidth = 1.5 * dpr;
      ctx.strokeStyle = 'rgba(32, 220, 255, 0.9)';
      ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
      ctx.setLineDash([]);
    }

    const box = drag ? [drag.x0, drag.y0, drag.x1, drag.y1] : subject.mode === 'box' ? subject.box : null;
    if (box) {
      const x = Math.min(box[0], box[2]) * sx;
      const y = Math.min(box[1], box[3]) * sy;
      const w = Math.abs(box[2] - box[0]) * sx;
      const hgt = Math.abs(box[3] - box[1]) * sy;
      ctx.fillStyle = 'rgba(0, 0, 0, 0.38)';
      ctx.beginPath();
      ctx.rect(0, 0, width, height);
      ctx.rect(x, y, w, hgt);
      ctx.fill('evenodd');
      ctx.lineWidth = 3 * dpr;
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.6)';
      ctx.strokeRect(x, y, w, hgt);
      ctx.lineWidth = 1.5 * dpr;
      ctx.strokeStyle = '#6aa0ff';
      ctx.strokeRect(x, y, w, hgt);
    }

    if (subject.mode === 'points') {
      for (const p of subject.points) {
        const cx = p.x * sx;
        const cy = p.y * sy;
        const r = POINT_RADIUS * dpr;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fillStyle = p.label === 1 ? '#2fbf71' : '#e5484d';
        ctx.fill();
        ctx.lineWidth = 2 * dpr;
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();
        ctx.beginPath();
        ctx.lineWidth = 2 * dpr;
        ctx.moveTo(cx - r * 0.5, cy);
        ctx.lineTo(cx + r * 0.5, cy);
        if (p.label === 1) {
          ctx.moveTo(cx, cy - r * 0.5);
          ctx.lineTo(cx, cy + r * 0.5);
        }
        ctx.stroke();
      }
    }
  }

  // ---------------------------------------------------------------- pointer input (display pixels)
  function toFrame(event) {
    const rect = canvas.getBoundingClientRect();
    const fx = clamp((event.clientX - rect.left) / rect.width, 0, 1);
    const fy = clamp((event.clientY - rect.top) / rect.height, 0, 1);
    return [fx * frameW(), fy * frameH(), rect.width ? frameW() / rect.width : 1];
  }

  canvas.addEventListener('pointerdown', (event) => {
    const subject = current();
    if (!image || imageIndex !== subject.frame_index) return;
    if (subject.mode === 'box' && event.button === 0) {
      const [x, y] = toFrame(event);
      drag = { x0: x, y0: y, x1: x, y1: y };
      canvas.setPointerCapture(event.pointerId);
      event.preventDefault();
      draw();
    } else if (subject.mode === 'points' && (event.button === 0 || event.button === 2)) {
      const [x, y, perCss] = toFrame(event);
      const hit = subject.points.findIndex((p) => Math.hypot(p.x - x, p.y - y) <= (POINT_RADIUS + 3) * perCss);
      if (hit >= 0) {
        commit({ points: subject.points.filter((_, index) => index !== hit) });
      } else {
        const label = event.button === 2 || negative ? 0 : 1;
        commit({ points: [...subject.points, { x: Math.round(x * 100) / 100, y: Math.round(y * 100) / 100, label }] });
      }
      event.preventDefault();
    }
  });
  canvas.addEventListener('pointermove', (event) => {
    if (!drag) return;
    const [x, y] = toFrame(event);
    drag.x1 = x;
    drag.y1 = y;
    draw();
  });
  const endDrag = (event) => {
    if (!drag) return;
    const d = drag;
    drag = null;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    const [x1, x2] = [Math.min(d.x0, d.x1), Math.max(d.x0, d.x1)];
    const [y1, y2] = [Math.min(d.y0, d.y1), Math.max(d.y0, d.y1)];
    // a click without a drag (or a tiny box) keeps the box that was there
    if (x2 - x1 < 4 || y2 - y1 < 4) {
      draw();
      return;
    }
    commit({ box: [x1, y1, x2, y2].map((v) => Math.round(v * 100) / 100) });
  };
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);

  function undo() {
    const previous = undoStack.pop();
    if (!previous) return;
    updateSubject(videoId, previous);
  }

  function clearMarks() {
    const subject = current();
    if (!subject.points.length && !subject.box) return;
    commit({ points: [], box: null });
  }

  scope.on(document, 'keydown', (event) => {
    if (!stage.isConnected || document.querySelector('dialog[open]')) return;
    if ((event.ctrlKey || event.metaKey) && !event.shiftKey && event.key.toLowerCase() === 'z' && !isEditableTarget(event.target)) {
      event.preventDefault();
      undo();
    }
  });
  if (typeof ResizeObserver === 'function') {
    const observer = new ResizeObserver(() => draw());
    observer.observe(stage);
    scope.add(() => observer.disconnect());
  }

  // ---------------------------------------------------------------- preview
  async function runPreview() {
    const state = store.get();
    const subject = current();
    const missing = subjectProblem(subject);
    if (missing) {
      previewError = { key: previewKey(), error: toApiError({ code: 'incomplete', message: missing }) };
      renderResult();
      return;
    }
    const key = previewKey();
    try {
      const result = await api.subjectPreview(videoId, subjectPayload(subject, state.options), config.get('model'));
      let overlay = null;
      if (result.overlay_png) {
        overlay = new Image();
        await new Promise((resolve) => {
          overlay.onload = resolve;
          overlay.onerror = resolve;
          overlay.src = result.overlay_png;
        });
      }
      if (scope.disposed) return;
      let shownKey = key;
      if (previewKey() === key && isNumber(result.frame_index) && result.frame_index !== subject.frame_index) {
        // The engine found that a mask picture shows another frame: follow it, so that the frame on
        // screen, the overlay and the job all use the same reference frame.
        commit({ frame_index: clamp(Math.round(result.frame_index), 0, total - 1) });
        shownKey = previewKey();
        toast({ type: 'info', title: `Reference frame moved to ${formatInt(result.frame_index)}`, text: 'The mask picture shows that frame of the video.', timeout: 6000 });
      }
      preview = { key: shownKey, result, overlay };
      previewError = null;
    } catch (error) {
      if (scope.disposed) return;
      previewError = { key, error: toApiError(error) };
    }
    renderResult();
    draw();
  }

  function renderResult() {
    const key = previewKey();
    if (previewError && previewError.key === key) {
      const err = previewError.error;
      let title = 'The preview failed';
      let tone = 'err';
      let extra = null;
      if (err.code === 'busy') {
        title = 'A job is being processed';
        tone = 'warn';
      } else if (err.code === 'prompt_not_supported' || err.code === 'model_not_installed' || err.code === 'weights_missing') {
        title = 'A helper model is needed for this preview';
        tone = 'warn';
        extra = [
          h('p', { class: 'field-hint' }, 'The selection is still saved and sent with the job; the job reports the same message if the helper is still missing when it starts.'),
          button({ label: 'Open Models', size: 'small', iconRight: 'arrow-right', onClick: () => navigate('models') }),
        ];
      } else if (err.code === 'incomplete') {
        title = 'The selection is not complete';
        tone = 'warn';
      }
      fill(resultSlot, errorNote(err, { title, tone, testid: 'subject-preview-error' }), extra ? h('div', { class: 'subj-extra' }, extra) : null);
      // errorNote puts the message under the title; the API message is the important part
      return;
    }
    if (preview && preview.key === key) {
      const r = preview.result;
      const backends = r.backends && typeof r.backends === 'object' ? Object.entries(r.backends) : [];
      fill(
        resultSlot,
        h(
          'div',
          { class: 'subj-facts', testid: 'subject-preview-facts' },
          r.has_mask
            ? note({ tone: 'ok', title: 'The selection resolves to a mask', text: 'It is drawn in cyan on the frame.' })
            : note({ tone: 'warn', title: 'No mask for this selection', text: 'The server accepted the selection but could not compute a mask preview for it (see the notes and the available helpers below). The job still receives the selection.' }),
          kv(
            [
              ['Resolved by', r.source || null],
              ['Score', isNumber(r.score) ? r.score.toFixed(3) : null],
              ['Area', isNumber(r.area_fraction) ? `${formatPercent(r.area_fraction * 100, 1)} of the frame` : null],
              ['Label', r.label || null],
              ['Frame', isNumber(r.frame_index) ? `${formatInt(r.frame_index)} (${r.width}×${r.height})` : null],
              ['Box', Array.isArray(r.box) ? r.box.map((v) => Math.round(v)).join(', ') : null],
            ],
            'subject-preview-kv',
          ),
          r.notes && r.notes.length ? h('div', null, h('div', { class: 'section-title' }, 'Notes'), bullets(r.notes)) : null,
          backends.length
            ? h(
                'div',
                { class: 'subj-backends' },
                h('span', { class: 'faint' }, 'Helpers:'),
                backends.map(([name, ok]) => badge(humanize(name), ok ? 'ok' : '', ok ? 'check' : 'minus', ok ? 'available' : 'not installed')),
              )
            : null,
        ),
      );
      return;
    }
    fill(resultSlot, h('p', { class: 'muted', testid: 'subject-preview-none' }, 'Preview selection asks the engine what this selection resolves to on the reference frame.'));
  }

  // ---------------------------------------------------------------- sync with the store
  function sync() {
    const subject = current();
    tools.set(subject.mode);
    polarity.el.hidden = subject.mode !== 'points';
    toolHelp.textContent = TOOL_HELP[subject.mode] || '';
    textField.hidden = subject.mode !== 'text';
    maskField.hidden = subject.mode !== 'mask';
    if (document.activeElement !== textInput) textInput.value = subject.text || '';
    if (document.activeElement !== maskInput) maskInput.value = subject.mask_path || '';
    if (document.activeElement !== slider) slider.value = String(subject.frame_index);
    frameLabel.textContent = `Frame ${formatInt(subject.frame_index)} / ${formatInt(total - 1)}`;
    isolate.set(subject.isolate === false);
    backward.set(subject.track_backward !== false);
    backward.setDisabled(subject.frame_index === 0);
    const counts = subject.mode === 'points' && subject.points.length ? ` (${plural(subject.points.length, 'marker')})` : '';
    summary.textContent = `${describeSubject(subject)}${counts}`;
    const missing = subjectProblem(subject);
    problem.hidden = !missing;
    problem.textContent = missing || '';
    undoButton.disabled = undoStack.length === 0;
    clearButton.hidden = !(subject.mode === 'points' || subject.mode === 'box');
    clearButton.disabled = !subject.points.length && !subject.box;
    resetButton.disabled = isDefaultSubject(subject, store.get().options) && !subject.points.length && !subject.box && !subject.text && !subject.mask_path;
    if (imageIndex !== subject.frame_index) requestFrame(subject.frame_index);
    draw();
    renderResult();
  }
  scope.add(store.watch((state) => [state.subjects[videoId], state.options, state.configVersion], sync));

  const frameCard = card({
    title: 'Reference frame',
    sub: dims ? `${dims[0]}×${dims[1]} · ${plural(total, 'frame')}` : plural(total, 'frame'),
    body: h('div', { class: 'stack' }, stage, scrubber),
    testid: 'subject-frame-card',
  });

  const toolCard = card({
    title: 'Selection',
    body: h(
      'div',
      { class: 'stack' },
      tools.el,
      polarity.el,
      toolHelp,
      textField,
      maskField,
      h('div', { class: 'subj-state' }, icon('target', 's'), summary),
      problem,
      h('div', { class: 'btn-row' }, previewButton, undoButton, clearButton, h('span', { class: 'grow' }), resetButton),
      resultSlot,
    ),
    testid: 'subject-tools',
  });

  const optionsCard = card({
    title: 'Options',
    body: h('div', { class: 'subj-options' }, isolate.el, backward.el),
    testid: 'subject-options',
  });

  return { main: [frameCard, optionsCard], side: [toolCard] };
}

// ------------------------------------------------------------------ explanations

function explanations(state) {
  const model = currentModel(state);
  const items = [];
  if (model) {
    const caps = model.capabilities || {};
    const native = caps.native_prompts || [];
    if (!caps.tracks_single_subject) {
      items.push(
        note({
          tone: 'info',
          title: `${model.name} has no built-in subject selection`,
          text: 'The model mattes every subject it recognises. Your selection is used to isolate the chosen subject by connected-component tracking from the reference frame on; this cannot separate subjects that touch or overlap - use “Keep every subject” or a model with subject selection for such shots.',
          testid: 'subject-explain-tracking',
        }),
      );
    }
    items.push(
      h(
        'p',
        { class: 'field-hint', testid: 'subject-explain-prompts' },
        native.length
          ? `${model.name} understands ${native.join(', ')} directly. Other selections are first turned into a mask by a helper model.`
          : `${model.name} does not read clicks, boxes or text itself: they are turned into a mask by a helper model (subject detector, segmentation or text grounding) before the matting runs.`,
      ),
    );
  }
  const helpers = (state.models || []).filter((m) => m.kind === 'auxiliary');
  const ready = helpers.filter((m) => m.status && m.status.ready);
  if (state.models && !ready.length) {
    items.push(
      note({
        tone: 'warn',
        title: helpers.length ? 'No helper model is ready' : 'No helper model is installed',
        text: 'Automatic detection and text descriptions need a helper model; the preview then reports what is missing. You can still set a box, clicks, a text or a mask path here: they are submitted with the job.',
        actions: [button({ label: 'Open Models', size: 'small', iconRight: 'arrow-right', onClick: () => navigate('models') })],
        testid: 'subject-explain-helpers',
      }),
    );
  }
  return items;
}

// ------------------------------------------------------------------ view

export function subjectView(scope) {
  const switcher = h('div', { class: 'subj-switch' });
  const body = h('div');
  const explain = h('div', { class: 'stack', testid: 'subject-explanations' });
  let editorFor = null;
  let editorScope = null;
  const releaseEditor = () => {
    if (editorScope) editorScope.dispose();
    editorScope = null;
    editorFor = null;
  };
  scope.add(releaseEditor);

  scope.add(
    store.watch(
      (state) => {
        const video = activeVideo(state);
        return [video ? video.id : null, state.videos === null, state.videosError, selectedVideos(state).map((v) => v.id).join(',')];
      },
      () => {
        const state = store.get();
        const video = activeVideo(state);
        const selected = selectedVideos(state);
        fill(
          switcher,
          selected.length > 1
            ? [
                h('span', { class: 'faint' }, 'Video:'),
                selected.map((item) =>
                  h(
                    'button',
                    { class: ['chip', item.id === (video && video.id) && 'is-on'], type: 'button', testid: 'subject-video', 'aria-pressed': String(item.id === (video && video.id)), onClick: () => setActive(item.id), tip: describeSubject(subjectFor(state, item.id)) },
                    h('span', { class: 'truncate' }, item.filename),
                  ),
                ),
                h('span', { class: 'field-hint' }, 'Every video of a batch has its own selection.'),
              ]
            : [],
        );
        if (!video) {
          releaseEditor();
          fill(
            body,
            state.videos === null
              ? state.videosError
                ? errorNote(state.videosError, { title: 'The library could not be loaded' })
                : skeletonCards(2)
              : card({
                  body: emptyState({
                    iconName: 'target',
                    title: 'No video selected',
                    text: 'Choose a video first; then pick the subject on one of its frames.',
                    actions: [button({ label: 'Choose a video', icon: 'film', variant: 'primary', onClick: () => navigate('source'), testid: 'subject-choose-video' })],
                    testid: 'subject-empty',
                  }),
                }),
          );
          return;
        }
        if (editorFor === video.id) return;
        releaseEditor();
        editorFor = video.id;
        editorScope = createScope();
        const editor = subjectEditor(editorScope, video);
        fill(body, h('div', { class: 'subject-layout' }, h('div', { class: 'stack' }, editor.main), h('aside', { class: 'stack subject-side' }, editor.side, explain)));
      },
    ),
  );

  scope.add(
    store.watch(
      (state) => [state.models, state.configVersion],
      () => fill(explain, explanations(store.get())),
    ),
  );

  return h(
    'div',
    { class: 'subject', testid: 'view-subject' },
    h(
      'div',
      { class: 'view-head' },
      h('div', { class: 'view-title' }, h('h1', null, 'Subject'), h('p', null, 'Choose who is kept. Leave it on Auto, or click the subject, draw a box, describe it or give a mask image of the reference frame.')),
      h(
        'div',
        { class: 'view-actions' },
        button({ label: 'Source', icon: 'chevron-left', onClick: () => navigate('source'), testid: 'subject-back' }),
        button({ label: 'Next: settings', iconRight: 'arrow-right', variant: 'primary', onClick: () => navigate('settings'), testid: 'subject-next' }),
      ),
    ),
    switcher,
    body,
  );
}
