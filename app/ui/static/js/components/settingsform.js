// The settings form, bound to a ConfigModel. The Settings view uses it for the configuration of
// new jobs; the re-render dialog of Results uses it with the stored configuration of a finished
// job as the baseline. Controls are created once and kept in step with the model by `sync()`, so
// typing and dragging are never interrupted by a re-render.

import { h, fill } from '../dom.js';
import { store, activeVideo, automaticVariant, variantReady } from '../state.js';
import {
  AUDIO_CODECS,
  BITRATE_PATTERN,
  CODEC_GROUPS,
  EXPORTS,
  MATTE_FINGERPRINT,
  QUALITY_RANGE,
  REFINEMENT_GROUPS,
  REFINEMENT_NEEDS,
  REFINEMENT_PRESET_LABELS,
  RESOLUTION_PRESETS,
  automaticContainer,
  codecInfo,
  containerChoices,
  effectiveRefinementPreset,
  encoderPresetChoices,
  pixelFormatChoices,
  plannedCodecId,
  presetRefinementValue,
  qualityInfo,
  reconcile,
} from '../config.js';
import { formatMb, isNumber } from '../format.js';
import { navigate } from '../actions.js';
import { button, card, checkbox, disclosure, errorNote, field, note, optionalNumber, optionalText, segmented, select, skeletonCards, sliderNumber, switchControl } from './controls.js';
import { capabilityFacts, installHint, licenseBlock, limitationsDisclosure, modelState, packageProblems, statusBadge, vramHint } from './modelinfo.js';

const CUSTOM_RESOLUTION = 'custom';
const HEX_COLOUR = /^#?[0-9a-fA-F]{6}$/;

// where the remark about an automatic adjustment is shown: changed path -> field that was adjusted
const ADJUSTMENT_TARGET = {
  model: 'variant',
  background: 'encoding.codec',
  'encoding.codec': 'encoding.container',
  'encoding.container': 'encoding.codec',
};

/**
 * Build the form for `model` (a ConfigModel).
 *   scope     clean-up scope of the caller (view or dialog)
 *   rerender  true in the re-render dialog: settings that make the model run again are marked
 * Returns {el, setErrors(fields), fingerprintChanges()}.
 */
export function createSettingsForm({ model, scope, rerender = false }) {
  const fields = new Map(); // settings path -> {api, reset, presetAware}
  // paths without a field of their own whose validation messages belong to another field
  const errorTargets = new Map(); // settings path -> field api
  const sections = []; // {sync(state, options)}
  let built = false;
  let noted = null; // field that carries the remark of the last automatic adjustment

  const body = h('div', { class: 'settings-form', testid: 'settings-form' });

  // ---------------------------------------------------------------- shared helpers

  function showAdjustments(path, notes) {
    if (noted) noted.setNote('');
    noted = null;
    if (!notes.length) return;
    const target = fields.get(ADJUSTMENT_TARGET[path] || path) || fields.get(path);
    if (!target) return;
    target.api.setNote(notes.join(' '));
    noted = target.api;
  }

  /** Apply a change made by the user and keep dependent settings consistent. */
  function change(path, value) {
    model.set(path, value);
    showAdjustments(path, reconcile(model, store.get().options, path));
  }

  /** `group`: the label names several controls (it must not act as the label of the first one). */
  function addField(path, { label, control, hint = '', wide = false, testid = null, presetAware = false, group = false }) {
    const api = field({ label, control, hint, path, wide, testid, labelFor: group ? false : null });
    const reset = button({ icon: 'undo', size: 'small', variant: 'ghost', tip: presetAware ? 'Back to the value of the preset' : 'Back to the default', testid: `reset-${path}` });
    reset.setAttribute('aria-label', `Reset: ${label}`);
    reset.hidden = true;
    reset.addEventListener('click', () => {
      model.reset(path);
      showAdjustments(path, reconcile(model, store.get().options, path));
    });
    api.setReset(reset);
    fields.set(path, { api, reset, presetAware });
    return api;
  }

  /** "changed" tags and reset buttons follow the model; fingerprint settings are marked in the re-render dialog. */
  function decorate() {
    const fingerprint = rerender ? new Set(MATTE_FINGERPRINT.map((item) => item.path)) : null;
    for (const [path, entry] of fields) {
      const changed = model.isChanged(path);
      entry.reset.hidden = !changed;
      if (changed) entry.api.setTag(fingerprint && fingerprint.has(path) ? 'changed · runs the model again' : 'changed', true);
      else if (entry.presetAware && model.get(path) === null) entry.api.setTag('preset', false);
      else entry.api.setTag('', false);
    }
  }

  function section({ title, sub = null, content, testid }) {
    return card({ title, sub, body: content, testid });
  }

  // ---------------------------------------------------------------- model

  function modelSection() {
    const list = h('div', { class: 'model-options', role: 'radiogroup', 'aria-label': 'Matting model', testid: 'model-options' });
    const stateSlot = h('div');
    const variantSelect = select({ options: [], value: null, label: 'Model variant', testid: 'variant-select', onChange: (value) => change('variant', value) });
    const variantField = addField('variant', { label: 'Variant', control: variantSelect.el, wide: true });
    let signature = null;
    let cards = new Map();

    const choose = (id) => {
      if (model.get('model') !== id) change('model', id);
    };

    function optionCard(item) {
      const state = modelState(item);
      const problems = packageProblems(item);
      const el = h(
        'div',
        { class: 'model-option', role: 'radio', tabIndex: -1, 'aria-checked': 'false', testid: `model-option-${item.id}`, dataset: { modelId: item.id, state: state.id } },
        h(
          'div',
          { class: 'model-option-head' },
          h('span', { class: 'radio-mark', 'aria-hidden': 'true' }),
          h('span', { class: 'model-option-name' }, item.name),
          h('span', { class: 'model-option-id mono' }, item.id),
          statusBadge(item),
        ),
        item.description ? h('p', { class: 'model-option-desc break' }, item.description) : null,
        capabilityFacts(item),
        licenseBlock(item),
        problems.length
          ? note({
              tone: 'err',
              iconName: 'terminal',
              title: 'Not installed: jobs with this model fail until this is fixed',
              children: problems.map((problem) => installHint(problem, item.id)),
              testid: 'model-not-installed',
            })
          : null,
        state.id === 'weights'
          ? note({
              tone: 'warn',
              iconName: 'download',
              title: 'The weight files are not on this computer yet',
              text: 'Download them in Models. A job downloads them itself only when “Download missing weights” allows it (Advanced, below).',
              actions: rerender ? null : [button({ label: 'Open Models', size: 'small', iconRight: 'arrow-right', onClick: () => navigate('models') })],
              testid: 'model-weights-missing',
            })
          : null,
        limitationsDisclosure(item),
      );
      el.addEventListener('click', (event) => {
        if (event.target.closest('a, button, summary')) return;
        choose(item.id);
      });
      el.addEventListener('keydown', (event) => {
        if (event.target !== el) return;
        const order = Array.from(cards.keys());
        const position = order.indexOf(item.id);
        if (event.key === ' ' || event.key === 'Enter') {
          event.preventDefault();
          choose(item.id);
        } else if (['ArrowDown', 'ArrowRight', 'ArrowUp', 'ArrowLeft'].includes(event.key)) {
          event.preventDefault();
          const step = event.key === 'ArrowDown' || event.key === 'ArrowRight' ? 1 : -1;
          const next = order[(position + step + order.length) % order.length];
          choose(next);
          cards.get(next).focus();
        }
      });
      return el;
    }

    const el = section({
      title: 'Model',
      sub: rerender ? 'changing it runs the model again' : null,
      content: h('div', { class: 'stack' }, stateSlot, list, h('div', { class: 'form-grid' }, variantField.el)),
      testid: 'settings-model',
    });

    function sync(state, options) {
      const models = state.models ? state.models.filter((item) => item.kind !== 'auxiliary') : null;
      const current = model.get('model');
      const nextSignature = JSON.stringify([models, state.modelsError ? state.modelsError.message : null]);
      if (nextSignature !== signature) {
        signature = nextSignature;
        cards = new Map();
        if (models) for (const item of models) cards.set(item.id, optionCard(item));
        fill(list, Array.from(cards.values()));
      }
      const known = !!models && models.some((item) => item.id === current);
      if (!models) {
        fill(stateSlot, state.modelsError ? errorNote(state.modelsError, { title: 'The model list could not be loaded', testid: 'settings-models-error' }) : skeletonCards(2));
      } else if (!models.length) {
        fill(stateSlot, note({ tone: 'warn', title: 'No matting model is registered in this installation.', testid: 'settings-no-models' }));
      } else if (!known) {
        fill(
          stateSlot,
          note({
            tone: 'err',
            title: `The selected model “${current}” does not exist in this installation.`,
            text: 'Choose one of the models below.',
            testid: 'settings-unknown-model',
          }),
        );
      } else {
        fill(stateSlot);
      }
      stateSlot.hidden = !stateSlot.firstChild;
      const focusable = known ? current : cards.keys().next().value;
      for (const [id, cardEl] of cards) {
        cardEl.setAttribute('aria-checked', String(id === current));
        cardEl.tabIndex = id === focusable ? 0 : -1;
      }

      const selected = known ? models.find((item) => item.id === current) : null;
      const variants = selected ? selected.variants || [] : [];
      variantField.el.hidden = variants.length < 2;
      if (selected && variants.length >= 2) {
        const auto = automaticVariant(selected, model.get('quality'));
        const quality = qualityInfo(options, model.get('quality'));
        variantSelect.setOptions(
          [
            { value: null, label: `Automatic${auto ? `: ${auto.label}` : ''}${quality ? ` (quality preset ${quality.label})` : ''}` },
            ...variants.map((variant) => ({
              value: variant.id,
              label: `${variant.label || variant.id}${variantReady(selected, variant.id) ? '' : ' - weights not downloaded'}`,
            })),
          ],
          model.get('variant'),
        );
        const effective = variants.find((variant) => variant.id === model.get('variant')) || auto;
        const parts = effective ? [effective.description || null, vramHint(effective)].filter(Boolean) : [];
        if (effective && !variantReady(selected, effective.id)) parts.push('Its weight files are not downloaded yet.');
        variantField.setHint(parts.join(' · '));
      }
    }
    return { el, sync };
  }

  // ---------------------------------------------------------------- quality preset

  function qualitySection(options) {
    const description = h('div', { class: 'field-hint', testid: 'quality-description' });
    const control = segmented({
      options: options.quality_presets.map((quality) => ({ value: quality.id, label: quality.label })),
      value: model.get('quality'),
      label: 'Quality preset',
      testid: 'quality',
      onChange: (value) => change('quality', value),
    });
    const qualityField = addField('quality', { label: 'Quality preset', control: h('div', { class: 'stack tight' }, control.el, description), wide: true, group: true });
    const el = section({
      title: 'Quality',
      sub: rerender ? 'changing it runs the model again' : null,
      content: h('div', { class: 'form-grid' }, qualityField.el),
      testid: 'settings-quality',
    });
    function sync(state, opts) {
      const quality = qualityInfo(opts, model.get('quality'));
      control.set(model.get('quality'));
      const refinement = quality ? REFINEMENT_PRESET_LABELS[quality.refinement_preset] || quality.refinement_preset : null;
      description.textContent = quality ? `${quality.description}${refinement ? ` Refinement preset: ${refinement}.` : ''}` : '';
    }
    return { el, sync };
  }

  // ---------------------------------------------------------------- hardware

  function hardwareSection() {
    const deviceSelect = select({ options: [], value: null, label: 'Device', testid: 'device-select', onChange: (value) => change('device', value) });
    const deviceField = addField('device', { label: 'Device', control: deviceSelect.el });
    const precision = segmented({
      options: [
        { value: 'auto', label: 'Automatic' },
        { value: 'fp32', label: 'Full (fp32)' },
        { value: 'fp16', label: 'Half (fp16)' },
      ],
      value: model.get('precision'),
      label: 'Precision',
      testid: 'precision',
      onChange: (value) => change('precision', value),
    });
    const precisionField = addField('precision', { label: 'Precision', control: precision.el, group: true });
    const batch = optionalNumber({ min: 1, max: 64, value: model.get('batch_size'), label: 'Batch size', testid: 'batch-size', onChange: (value) => change('batch_size', value) });
    const batchField = addField('batch_size', { label: 'Batch size', control: batch.el });

    const el = section({ title: 'Hardware', content: h('div', { class: 'form-grid' }, deviceField.el, precisionField.el, batchField.el), testid: 'settings-hardware' });

    function sync(state) {
      const system = state.system;
      const current = model.get('device');
      const list = [{ value: 'auto', label: system ? `Automatic (${system.default_device})` : 'Automatic' }];
      for (const device of system ? system.devices || [] : []) {
        const memory = isNumber(device.total_memory_mb)
          ? ` · ${isNumber(device.free_memory_mb) ? `${formatMb(device.free_memory_mb)} free of ` : ''}${formatMb(device.total_memory_mb)} ${device.type === 'cpu' ? 'RAM' : 'VRAM'}`
          : '';
        list.push({ value: device.id, label: `${device.id} · ${device.name}${memory}${device.available ? '' : ' (not usable)'}`, disabled: !device.available });
      }
      if (!list.some((option) => option.value === current)) {
        const cudaAlias = current === 'cuda' && list.some((option) => option.value === 'cuda:0');
        list.push({ value: current, label: cudaAlias ? 'cuda (the first CUDA device)' : `${current} (not detected on this computer)` });
      }
      deviceSelect.setOptions(list, current);
      const unusable = system ? (system.devices || []).find((device) => device.id === current && !device.available) : null;
      deviceField.setHint(
        unusable
          ? `Not usable: ${unusable.detail}`
          : current === 'auto'
            ? 'The best available accelerator; the CPU when none can be initialised.'
            : state.systemError && !system
              ? 'The device list could not be loaded.'
              : '',
      );

      precision.set(model.get('precision'));
      const selected = (state.models || []).find((item) => item.id === model.get('model')) || null;
      const noHalf = selected && selected.capabilities && selected.capabilities.supports_fp16 === false;
      precisionField.setHint(
        noHalf
          ? `${selected.name} does not support half precision: fp32 is used.`
          : 'Half precision roughly halves the memory use; it is only used on CUDA. Automatic picks it when the model supports it.',
      );

      batch.set(model.get('batch_size'));
      const limit = selected && selected.capabilities ? selected.capabilities.max_batch_size : null;
      batchField.setHint(
        `Frames per inference step (1-64). Automatic follows the frame size and the free memory, and halves the batch when memory runs out.${limit ? ` ${selected.name} processes at most ${limit} per step.` : ''}`,
      );
    }
    return { el, sync };
  }

  // ---------------------------------------------------------------- background

  function backgroundSection(options) {
    const group = h('div', { class: 'swatches', role: 'radiogroup', 'aria-label': 'Background', testid: 'background-options' });
    const colorInput = h('input', { type: 'color', class: 'color-input', 'aria-label': 'Custom background colour', testid: 'background-color' });
    const hexInput = h('input', { class: 'input mono', type: 'text', maxLength: 7, placeholder: '#00FF00', spellcheck: false, autocomplete: 'off', 'aria-label': 'Custom background colour as #RRGGBB', testid: 'background-hex' });
    const customRow = h('div', { class: 'input-group swatch-custom', hidden: true }, colorInput, hexInput);
    const buttons = new Map();

    for (const background of options.backgrounds) {
      const swatch = h('span', { class: ['swatch', background.id === 'transparent' && 'checker'], style: background.color ? { '--c': background.color } : null });
      const item = h(
        'button',
        {
          type: 'button',
          class: 'swatch-option',
          role: 'radio',
          'aria-checked': 'false',
          testid: `background-${background.id}`,
          dataset: { value: background.id },
          onClick: () => {
            if (model.get('background') !== background.id) change('background', background.id);
          },
        },
        swatch,
        h('span', null, background.label),
        background.color ? h('span', { class: 'swatch-code mono' }, background.color) : null,
      );
      buttons.set(background.id, { item, swatch });
      group.appendChild(item);
    }
    group.addEventListener('keydown', (event) => {
      if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
      const order = Array.from(buttons.keys());
      const position = order.indexOf(model.get('background'));
      const step = event.key === 'ArrowLeft' || event.key === 'ArrowUp' ? -1 : 1;
      const next = order[(Math.max(position, 0) + step + order.length) % order.length];
      event.preventDefault();
      change('background', next);
      buttons.get(next).item.focus();
    });

    colorInput.addEventListener('input', () => change('background_color', colorInput.value.toUpperCase()));
    hexInput.addEventListener('input', () => {
      const text = hexInput.value.trim();
      const ok = HEX_COLOUR.test(text);
      hexInput.classList.toggle('is-invalid', !ok && text !== '');
      if (ok) change('background_color', `#${text.replace('#', '').toUpperCase()}`);
    });
    hexInput.addEventListener('blur', () => {
      hexInput.classList.remove('is-invalid');
      hexInput.value = model.get('background_color') || '';
    });

    const backgroundField = addField('background', { label: 'Background', control: h('div', { class: 'stack tight' }, group, customRow), wide: true, group: true });
    // validation messages of the API about the colour belong to this field as well
    errorTargets.set('background_color', backgroundField);
    const el = section({ title: 'Background', content: h('div', { class: 'form-grid' }, backgroundField.el), testid: 'settings-background' });

    function sync() {
      const current = model.get('background');
      const color = model.get('background_color');
      for (const [id, entry] of buttons) {
        entry.item.setAttribute('aria-checked', String(id === current));
        entry.item.tabIndex = id === current ? 0 : -1;
      }
      const custom = buttons.get('custom');
      if (custom) custom.swatch.style.setProperty('--c', current === 'custom' && color ? color : 'transparent');
      customRow.hidden = current !== 'custom';
      if (current === 'custom' && color) {
        if (document.activeElement !== colorInput) colorInput.value = color.toLowerCase();
        if (document.activeElement !== hexInput) hexInput.value = color;
      }
      backgroundField.setHint(
        current === 'transparent'
          ? 'Transparent output is written with an alpha channel: only codecs that have one can be chosen below.'
          : current === 'custom'
            ? 'Any colour as #RRGGBB.'
            : 'The exact colour is written to every background pixel. Green is pure chroma green #00FF00.',
      );
    }
    return { el, sync };
  }

  // ---------------------------------------------------------------- matte refinement

  function refinementSection(options) {
    const presetControl = segmented({
      options: [{ value: null, label: 'By quality preset' }, ...Object.keys(options.refinement_presets).map((name) => ({ value: name, label: REFINEMENT_PRESET_LABELS[name] || name }))],
      value: model.get('refinement.preset'),
      label: 'Refinement preset',
      testid: 'refinement-preset',
      onChange: (value) => change('refinement.preset', value),
    });
    const presetField = addField('refinement.preset', { label: 'Preset', control: presetControl.el, wide: true, group: true });
    const controls = []; // {item, api, control, path}

    const groups = REFINEMENT_GROUPS.map((group) => {
      const items = group.fields.map((item) => {
        const path = `refinement.${item.key}`;
        let control;
        let api;
        if (item.type === 'bool') {
          // the field label names the tool; the text beside the switch says what it does
          control = switchControl({ label: item.hint, testid: `refine-${item.key}`, onChange: (value) => change(path, value) });
          api = addField(path, { label: item.label, control: control.el, presetAware: true });
        } else if (item.type === 'enum') {
          control = select({ options: item.options, value: null, label: item.label, testid: `refine-${item.key}`, onChange: (value) => change(path, value) });
          api = addField(path, { label: item.label, control: control.el, hint: item.hint, presetAware: true });
        } else {
          control = sliderNumber({
            min: item.min,
            max: item.max,
            step: item.step,
            soft: item.soft || null,
            scale: item.scale || 1,
            digits: item.digits,
            noSlider: !!item.noSlider,
            label: item.label,
            testid: `refine-${item.key}`,
            onChange: (value) => change(path, value),
          });
          api = addField(path, { label: item.unit ? `${item.label} (${item.unit})` : item.label, control: control.el, hint: item.hint, presetAware: true });
        }
        controls.push({ item, api, control, path });
        return api.el;
      });
      return h('div', { class: 'refine-group' }, h('div', { class: 'section-title' }, group.title), h('div', { class: 'form-grid' }, items));
    });

    const summary = h('span', null, 'Individual controls');
    const advanced = disclosure(summary, h('div', { class: 'stack loose' }, groups), { testid: 'refinement-advanced' });
    const offNote = note({ tone: 'info', text: 'Refinement is switched off: the matte of the model is used as it is. Changing a control below switches that tool on again.', testid: 'refinement-off' });
    const el = section({
      title: 'Matte refinement',
      content: h('div', { class: 'stack' }, h('div', { class: 'form-grid' }, presetField.el), offNote, advanced),
      testid: 'settings-refinement',
    });

    function sync(state, opts) {
      presetControl.set(model.get('refinement.preset'));
      const name = effectiveRefinementPreset(model, opts);
      const quality = qualityInfo(opts, model.get('quality'));
      presetField.setHint(
        model.get('refinement.preset') === null && name
          ? `In effect: ${REFINEMENT_PRESET_LABELS[name] || name}${quality ? `, as set by the quality preset ${quality.label}` : ''}. Controls you do not touch follow the preset.`
          : 'Controls you do not touch follow the preset.',
      );
      const effective = (key) => {
        const own = model.get(`refinement.${key}`);
        return own !== null ? own : presetRefinementValue(model, opts, key);
      };
      let overridden = 0;
      for (const entry of controls) {
        const own = model.get(entry.path);
        const preset = own === null;
        if (!preset) overridden += 1;
        const value = effective(entry.item.key);
        if (entry.item.type === 'enum') entry.control.set(value);
        else entry.control.set(value, { preset });
        const needs = REFINEMENT_NEEDS[entry.item.key];
        const active = needs ? needs(effective) : true;
        if (entry.control.setDisabled) entry.control.setDisabled(!active);
        else entry.control.el.disabled = !active;
        entry.api.el.classList.toggle('is-inactive', !active);
      }
      summary.textContent = overridden ? `Individual controls (${overridden} changed)` : 'Individual controls';
      offNote.hidden = !(name === 'off' && overridden === 0);
    }
    return { el, sync };
  }

  // ---------------------------------------------------------------- output (video)

  function outputSection() {
    const codecSelect = select({ options: [], value: null, label: 'Codec', testid: 'codec-select', onChange: (value) => change('encoding.codec', value) });
    const codecField = addField('encoding.codec', { label: 'Codec', control: codecSelect.el });
    const containerSelect = select({ options: [], value: null, label: 'Container', testid: 'container-select', onChange: (value) => change('encoding.container', value) });
    const containerField = addField('encoding.container', { label: 'Container', control: containerSelect.el });

    const crf = optionalNumber({ min: 0, max: 51, label: 'Constant quality', testid: 'crf', onChange: (value) => change('encoding.crf', value) });
    const crfField = addField('encoding.crf', { label: 'Constant quality (CRF)', control: crf.el });
    const bitrate = optionalText({ label: 'Target bitrate', placeholder: 'Automatic (constant quality)', testid: 'bitrate', mono: true, onChange: (value) => change('encoding.bitrate', value) });
    const bitrateField = addField('encoding.bitrate', { label: 'Target bitrate', control: bitrate.el });
    const maxBitrate = optionalText({ label: 'Maximum bitrate', placeholder: 'No limit', testid: 'max-bitrate', mono: true, onChange: (value) => change('encoding.max_bitrate', value) });
    const maxBitrateField = addField('encoding.max_bitrate', { label: 'Maximum bitrate', control: maxBitrate.el });
    const presetSelect = select({ options: [], value: null, label: 'Encoder preset', testid: 'encoder-preset', onChange: (value) => change('encoding.preset', value) });
    const presetField = addField('encoding.preset', { label: 'Encoder preset', control: presetSelect.el });
    const pixSelect = select({ options: [], value: null, label: 'Pixel format', testid: 'pixfmt-select', onChange: (value) => change('encoding.pix_fmt', value) });
    const pixField = addField('encoding.pix_fmt', { label: 'Pixel format', control: pixSelect.el });

    const resolutionSelect = select({
      options: [],
      value: null,
      label: 'Resolution',
      testid: 'resolution-select',
      onChange: (value) => {
        if (value !== CUSTOM_RESOLUTION) {
          change('encoding.resolution', value);
          return;
        }
        // start the custom size from the coded size of the selected video (or Full HD without one)
        const video = activeVideo(store.get());
        const stream = video && video.metadata ? video.metadata.video : null;
        change('encoding.resolution', stream && stream.width && stream.height ? `${stream.width}x${stream.height}` : '1920x1080');
      },
    });
    // the custom size is only written to the settings while both numbers are valid
    const customSize = { width: null, height: null };
    const applyCustom = () => {
      if (customSize.width !== null && customSize.height !== null) change('encoding.resolution', `${customSize.width}x${customSize.height}`);
    };
    const widthInput = optionalNumber({
      min: 16,
      max: 99999,
      placeholder: 'Width',
      label: 'Output width in pixels',
      testid: 'resolution-width',
      onChange: (value) => {
        customSize.width = value;
        applyCustom();
      },
    });
    const heightInput = optionalNumber({
      min: 16,
      max: 99999,
      placeholder: 'Height',
      label: 'Output height in pixels',
      testid: 'resolution-height',
      onChange: (value) => {
        customSize.height = value;
        applyCustom();
      },
    });
    const customRow = h('div', { class: 'input-group', hidden: true }, widthInput.el, h('span', { class: 'input-suffix' }, '×'), heightInput.el, h('span', { class: 'input-suffix' }, 'px'));
    const resolutionField = addField('encoding.resolution', { label: 'Resolution', control: h('div', { class: 'stack tight' }, resolutionSelect.el, customRow) });

    const noRateText = h('span');
    const noRateControl = note({ tone: 'info', children: noRateText, testid: 'no-rate-control' });
    const el = section({
      title: 'Output video',
      content: h(
        'div',
        { class: 'stack' },
        h('div', { class: 'form-grid' }, codecField.el, containerField.el, resolutionField.el, pixField.el),
        h('div', { class: 'section-title' }, 'Quality and size'),
        noRateControl,
        h('div', { class: 'form-grid' }, crfField.el, bitrateField.el, maxBitrateField.el, presetField.el),
      ),
      testid: 'settings-output',
    });

    function sync(state, opts) {
      const transparent = model.get('background') === 'transparent';
      const codecId = model.get('encoding.codec');
      const codec = codecInfo(opts, codecId);
      const planned = plannedCodecId(model);
      const plannedInfo = codecInfo(opts, planned);

      // codec: grouped by purpose; transparent output lists only codecs with an alpha channel
      const list = [{ value: null, label: 'Automatic' }];
      for (const group of CODEC_GROUPS) {
        for (const item of opts.codecs) {
          if (item.group !== group.id) continue;
          if (transparent && !item.supports_alpha) continue;
          list.push({ value: item.id, label: `${item.label}${item.available ? '' : ' - not in this FFmpeg build'}`, group: group.label, disabled: !item.available });
        }
      }
      if (codecId && !list.some((option) => option.value === codecId)) list.push({ value: codecId, label: codec ? codec.label : codecId });
      codecSelect.setOptions(list, codecId);
      codecField.setHint(
        codec
          ? `${codec.description}${codec.available ? '' : ' This FFmpeg build has no encoder for it: the job would fail.'}`
          : transparent
            ? 'Automatic: ProRes 4444 for .mov, VP9 for .webm, FFV1 for .mkv.'
            : 'Automatic: H.264 (H.265 for sources deeper than 8 bits), VP9 for .webm.',
      );

      // container: only those the codec can be stored in
      const auto = automaticContainer(model, opts);
      const containers = containerChoices(model, opts);
      const container = model.get('encoding.container');
      const containerList = [{ value: null, label: `Automatic (.${auto})` }, ...containers.map((name) => ({ value: name, label: `.${name}` }))];
      if (container && !containers.includes(container)) containerList.push({ value: container, label: `.${container} (not possible with this codec)` });
      containerSelect.setOptions(containerList, container);

      // resolution
      const resolution = model.get('encoding.resolution');
      const isPreset = resolution === 'source' || RESOLUTION_PRESETS.includes(resolution);
      const isLines = /^\d+p$/.test(resolution || '');
      const resolutionList = [{ value: 'source', label: 'Original (no scaling)' }, ...RESOLUTION_PRESETS.map((name) => ({ value: name, label: name }))];
      if (isLines && !isPreset) resolutionList.push({ value: resolution, label: resolution });
      resolutionList.push({ value: CUSTOM_RESOLUTION, label: 'Custom size' });
      const custom = !isPreset && !isLines;
      resolutionSelect.setOptions(resolutionList, custom ? CUSTOM_RESOLUTION : resolution);
      customRow.hidden = !custom;
      if (custom) {
        const match = String(resolution).match(/^(\d+)x(\d+)$/);
        customSize.width = match ? Number(match[1]) : null;
        customSize.height = match ? Number(match[2]) : null;
        widthInput.set(customSize.width);
        heightInput.set(customSize.height);
      }
      resolutionField.setHint(
        custom
          ? 'The picture is scaled to fit and padded with the background colour.'
          : resolution === 'source'
            ? 'The output keeps the resolution of the source.'
            : 'The short side gets this many pixels; the aspect ratio is kept.',
      );

      // pixel format
      const pixChoices = pixelFormatChoices(planned, transparent);
      const pix = model.get('encoding.pix_fmt');
      const pixList = [{ value: null, label: 'Automatic' }, ...pixChoices];
      if (pix && !pixChoices.some((choice) => choice.value === pix)) pixList.push({ value: pix, label: pix });
      pixSelect.setOptions(pixList, pix);
      pixField.setHint(
        pixChoices.length
          ? 'Automatic keeps the format of the source when the codec supports it.'
          : 'This codec defines the pixel format itself; automatic chooses the layout closest to the source.',
      );

      // rate control
      const range = QUALITY_RANGE[planned] || null;
      const hasRateControl = !!range;
      noRateControl.hidden = hasRateControl;
      if (!hasRateControl) {
        noRateText.textContent =
          plannedInfo && plannedInfo.lossless
            ? `${plannedInfo.label} is lossless: there is no quality setting.`
            : `${plannedInfo ? plannedInfo.label : 'This codec'} has a fixed quality profile: there is no quality setting.`;
      }
      for (const item of [crfField, bitrateField, maxBitrateField]) item.el.hidden = !hasRateControl;
      if (range) crf.setRange(range[0], range[1]);
      crf.set(model.get('encoding.crf'));
      const cq = plannedInfo && plannedInfo.rate_control === 'cq';
      crfField.el.querySelector('.field-label').textContent = cq ? 'Constant quality (CQ)' : 'Constant quality (CRF)';
      const bitrateValue = model.get('encoding.bitrate');
      const maxValue = model.get('encoding.max_bitrate');
      crfField.setHint(
        bitrateValue
          ? 'Not used while a target bitrate is set.'
          : range
            ? `${range[0]}-${range[1]}; lower is better quality and a larger file. Empty: chosen by the quality preset.`
            : '',
      );
      bitrate.set(bitrateValue);
      maxBitrate.set(maxValue);
      const badBitrate = !!bitrateValue && !BITRATE_PATTERN.test(bitrateValue);
      const badMax = !!maxValue && !BITRATE_PATTERN.test(maxValue);
      bitrate.setInvalid(badBitrate);
      maxBitrate.setInvalid(badMax);
      bitrateField.setHint(badBitrate ? 'Write it like 8000k, 20M or 5000000.' : 'For example 20M or 8000k. Replaces constant quality.');
      maxBitrateField.setHint(badMax ? 'Write it like 8000k, 20M or 5000000.' : 'Caps peaks of the bit rate, for example 30M.');

      const presets = encoderPresetChoices(planned);
      presetField.el.hidden = presets.length === 0;
      const preset = model.get('encoding.preset');
      const presetList = [{ value: null, label: 'Automatic (by quality preset)' }, ...presets];
      if (preset && !presets.some((choice) => choice.value === preset)) presetList.push({ value: preset, label: preset });
      presetSelect.setOptions(presetList, preset);
      presetField.setHint(planned === 'vp9' ? 'Encoding speed of libvpx.' : 'Slower presets compress better at the same quality.');
    }
    return { el, sync };
  }

  // ---------------------------------------------------------------- audio

  function audioSection() {
    const keep = switchControl({
      label: 'Keep the original audio',
      description: 'Audio streams are carried over to the output.',
      testid: 'preserve-audio',
      onChange: (value) => change('encoding.preserve_audio', value),
    });
    const keepField = addField('encoding.preserve_audio', { label: 'Audio', control: keep.el });
    const codecSelect = select({ options: AUDIO_CODECS, value: model.get('encoding.audio_codec'), label: 'Audio codec', testid: 'audio-codec', onChange: (value) => change('encoding.audio_codec', value) });
    const codecField = addField('encoding.audio_codec', { label: 'Audio codec', control: codecSelect.el });
    // an empty box means "the default bit rate" (shown as the placeholder), not "no value"
    const bitrate = optionalText({ label: 'Audio bitrate', placeholder: '', testid: 'audio-bitrate', mono: true, onChange: (value) => change('encoding.audio_bitrate', value || model.base('encoding.audio_bitrate')) });
    const bitrateField = addField('encoding.audio_bitrate', { label: 'Audio bitrate', control: bitrate.el, hint: 'Used when the audio is encoded again (AAC, Opus).' });
    const noAudio = note({ tone: 'info', text: 'The selected video has no audio stream: these settings have no effect for it.', testid: 'no-audio' });

    const el = section({
      title: 'Audio',
      content: h('div', { class: 'stack' }, noAudio, h('div', { class: 'form-grid' }, keepField.el, codecField.el, bitrateField.el)),
      testid: 'settings-audio',
    });

    function sync(state) {
      const preserve = model.get('encoding.preserve_audio') !== false;
      keep.set(preserve);
      const codec = model.get('encoding.audio_codec');
      codecSelect.set(codec);
      codecSelect.el.disabled = !preserve;
      codecField.setHint(
        codec === 'auto'
          ? 'The original stream is copied bit for bit when the container can hold it; otherwise it is encoded again and the report says so.'
          : codec === 'copy'
            ? 'Bit-for-bit copy. The job reports a problem when the container cannot hold the original codec.'
            : 'The audio is encoded again.',
      );
      const lossy = codec === 'aac' || codec === 'opus' || codec === 'auto';
      bitrateField.el.hidden = !preserve || !lossy;
      const value = model.get('encoding.audio_bitrate');
      bitrate.setPlaceholder(String(model.base('encoding.audio_bitrate') || ''));
      bitrate.set(model.isChanged('encoding.audio_bitrate') ? value : null);
      bitrate.setInvalid(!!value && !BITRATE_PATTERN.test(value));
      const video = rerender ? null : activeVideo(state);
      noAudio.hidden = !(video && video.metadata && video.metadata.has_audio === false);
    }
    return { el, sync };
  }

  // ---------------------------------------------------------------- additional deliverables

  function exportsSection() {
    const rows = EXPORTS.map((item) => {
      const path = `exports.${item.key}`;
      const remark = h('div', { class: 'field-hint', hidden: true });
      const control = checkbox({ label: item.label, description: item.hint, testid: `export-${item.key}`, onChange: (value) => change(path, value) });
      return { item, path, control, remark, el: h('div', { class: 'export-row' }, control.el, remark) };
    });
    const bitDepth = segmented({
      options: [
        { value: 8, label: '8-bit' },
        { value: 16, label: '16-bit' },
      ],
      value: model.get('exports.png_bit_depth'),
      label: 'PNG bit depth',
      testid: 'png-bit-depth',
      onChange: (value) => change('exports.png_bit_depth', value),
    });
    const bitDepthField = addField('exports.png_bit_depth', { label: 'PNG bit depth', control: bitDepth.el, hint: '16-bit keeps the precision of sources deeper than 8 bits.', group: true });
    const alphaMode = segmented({
      options: [
        { value: 'straight', label: 'Straight' },
        { value: 'premultiplied', label: 'Premultiplied' },
      ],
      value: model.get('exports.alpha_mode'),
      label: 'Alpha mode',
      testid: 'alpha-mode',
      onChange: (value) => change('exports.alpha_mode', value),
    });
    const alphaField = addField('exports.alpha_mode', { label: 'Alpha mode', control: alphaMode.el, hint: 'How colour and alpha are stored in transparent files. Straight is what most editing software expects.', group: true });
    // one field for the whole list: a tick is undone by clearing it, so the rows need no reset buttons
    const listField = field({ label: 'Additional files', control: h('div', { class: 'export-list' }, rows.map((row) => row.el)), path: 'exports', wide: true, labelFor: false });
    errorTargets.set('exports', listField);

    const el = section({
      title: 'Additional deliverables',
      content: h('div', { class: 'form-grid' }, listField.el, bitDepthField.el, alphaField.el),
      testid: 'settings-exports',
    });

    function sync(state) {
      const background = model.get('background');
      const color = (model.get('background_color') || '').toUpperCase();
      const transparent = background === 'transparent';
      const video = rerender ? null : activeVideo(state);
      const remarks = {
        green_screen: background === 'green' || (background === 'custom' && color === '#00FF00') ? 'The main output already is the green-screen version: no extra file is written.' : '',
        prores_4444: transparent && plannedCodecId(model) === 'prores_4444' ? 'The main output already is a transparent ProRes 4444 file: no extra file is written.' : '',
        audio: video && video.metadata && video.metadata.has_audio === false ? 'The selected video has no audio stream.' : '',
      };
      let changed = 0;
      for (const row of rows) {
        row.control.set(model.get(row.path) === true);
        if (model.isChanged(row.path)) changed += 1;
        const text = model.get(row.path) === true ? remarks[row.item.key] || '' : '';
        row.remark.hidden = !text;
        row.remark.textContent = text;
      }
      listField.setTag(changed ? 'changed' : '', changed > 0);
      bitDepth.set(model.get('exports.png_bit_depth'));
      alphaMode.set(model.get('exports.alpha_mode'));
      bitDepthField.el.hidden = model.get('exports.png_sequence') !== true;
      alphaField.el.hidden = !(transparent || model.get('exports.png_sequence') === true || model.get('exports.prores_4444') === true);
    }
    return { el, sync };
  }

  // ---------------------------------------------------------------- advanced

  function advancedSection() {
    const maxFrames = optionalNumber({ min: 1, max: null, placeholder: 'All frames', label: 'Frame limit', testid: 'max-frames', onChange: (value) => change('max_frames', value) });
    const maxFramesField = addField('max_frames', { label: 'Frame limit (quick draft)', control: maxFrames.el });

    const flag = (path, label, description, testid) => {
      const control = switchControl({ label: description, testid, onChange: (value) => change(path, value) });
      const api = addField(path, { label, control: control.el });
      return { control, api, path };
    };
    const flags = [
      flag('strict_verification', 'Strict verification', 'A failed quality check fails the job; the file is kept with a “.rejected” suffix for inspection.', 'strict-verification'),
      flag('allow_cpu_fallback', 'Fall back to the CPU', 'Continue on the CPU when the GPU cannot be initialised or runs out of memory at the smallest batch.', 'cpu-fallback'),
      flag('tolerate_decode_errors', 'Skip undecodable frames', 'Damaged frames are skipped and listed in the report instead of stopping the job.', 'tolerate-decode-errors'),
    ];

    const threeWay = (path, label, on, off, hint, testid) => {
      const control = segmented({
        options: [
          { value: null, label: 'Installation default' },
          { value: true, label: on },
          { value: false, label: off },
        ],
        value: model.get(path),
        label,
        testid,
        onChange: (value) => change(path, value),
      });
      const api = addField(path, { label, control: control.el, hint, group: true });
      return { control, api, path };
    };
    const choices = [
      threeWay('commercial_use', 'Commercial use', 'Commercial', 'Non-commercial', 'Commercial: models whose licence prohibits commercial use are refused. The installation default is set in the configuration file.', 'commercial-use'),
      threeWay('download_models', 'Download missing weights', 'Allow', 'Never', 'Whether a job may download missing weight files itself. The installation default is set in the configuration file.', 'download-models'),
    ];

    const el = section({
      title: 'Advanced',
      sub: rerender ? 'a frame limit runs the model again' : null,
      content: h('div', { class: 'form-grid' }, maxFramesField.el, choices.map((item) => item.api.el), flags.map((item) => item.api.el)),
      testid: 'settings-advanced',
    });

    function sync(state) {
      maxFrames.set(model.get('max_frames'));
      const video = rerender ? null : activeVideo(state);
      const count = video && video.metadata ? video.metadata.frame_count : null;
      maxFramesField.setHint(`Only the first N frames are processed, to judge the settings quickly. The output is then shorter than the source.${isNumber(count) ? ` The selected video has ${count} frames.` : ''}`);
      for (const item of flags) item.control.set(model.get(item.path) === true);
      for (const item of choices) item.control.set(model.get(item.path));
    }
    return { el, sync };
  }

  // ---------------------------------------------------------------- assembly

  function build(options) {
    built = true;
    const parts = [modelSection(), qualitySection(options), hardwareSection(), backgroundSection(options), refinementSection(options), outputSection(), audioSection(), exportsSection(), advancedSection()];
    sections.push(...parts);
    fill(body, parts.map((part) => part.el));
  }

  function sync() {
    const state = store.get();
    const options = state.options;
    if (!options || !model.ready) {
      if (!built) {
        fill(body, state.optionsError ? errorNote(state.optionsError, { title: 'The settings could not be loaded from the server', testid: 'settings-error' }) : skeletonCards(3));
      }
      return;
    }
    if (!built) build(options);
    for (const part of sections) part.sync(state, options);
    decorate();
  }

  scope.add(model.subscribe(sync));
  scope.add(
    store.watch(
      (state) => [state.options, state.optionsError, state.models, state.modelsError, state.system, state.systemError, state.activeId, state.selectedIds, state.videos],
      sync,
    ),
  );

  return {
    el: body,
    /**
     * Show validation messages of the API next to their controls. `messages` maps settings paths to
     * text; returns the entries no control exists for, so the caller can show them in a banner.
     */
    setErrors(messages) {
      const unplaced = [];
      const shown = new Set();
      const target = (path) => (fields.has(path) ? fields.get(path).api : errorTargets.get(path) || null);
      for (const entry of fields.values()) entry.api.setError('');
      for (const api of errorTargets.values()) api.setError('');
      for (const [path, text] of Object.entries(messages || {})) {
        // the field of the path itself, or of the nearest enclosing section ("exports.x" -> "exports")
        let key = path;
        while (key && !target(key)) key = key.includes('.') ? key.slice(0, key.lastIndexOf('.')) : '';
        const api = key ? target(key) : null;
        if (!api) {
          unplaced.push({ path, text });
          continue;
        }
        const previous = shown.has(api) ? `${api.el.querySelector('.field-error').textContent} ` : '';
        api.setError(previous + text);
        shown.add(api);
        const closed = api.el.closest('details:not([open])');
        if (closed) closed.open = true;
      }
      return unplaced;
    },
    /** Changed settings that make a re-render run the model again: [{path, label}]. */
    fingerprintChanges() {
      return MATTE_FINGERPRINT.filter((item) => model.isChanged(item.path));
    },
    /** Scroll to the first field that shows an error. */
    revealFirstError() {
      const first = body.querySelector('.field.has-error');
      if (first) first.scrollIntoView({ block: 'center', behavior: 'smooth' });
    },
  };
}
