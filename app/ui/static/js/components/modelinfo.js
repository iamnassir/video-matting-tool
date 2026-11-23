// How a model is presented to the user: installation state, capabilities, licence, limitations
// and install hints. Shared by the Models view (management) and the Settings view (selection),
// so both always describe a model in the same words.

import { h } from '../dom.js';
import { icon } from '../icons.js';
import { formatMb } from '../format.js';
import { badge, bullets, copyButton, disclosure } from './controls.js';

// ------------------------------------------------------------------ installation state

/**
 * The state of a model as GET /api/models reports it:
 *   'package'  a Python package (of the model or of a model it builds on) is not installed
 *   'ready'    at least one variant can be used right now
 *   'weights'  the package is there, the weight files are not
 */
export function modelState(model) {
  const status = (model && model.status) || {};
  if (status.installed === false) return { id: 'package', label: 'Package missing', tone: 'err', icon: 'terminal' };
  if (status.ready) return { id: 'ready', label: 'Ready', tone: 'ok', icon: 'check' };
  return { id: 'weights', label: 'Weights missing', tone: 'warn', icon: 'download' };
}

export function statusBadge(model) {
  const state = modelState(model);
  const tips = {
    package: 'A Python package this model needs is not installed',
    ready: 'Installed and ready to use',
    weights: 'The weight files have not been downloaded yet',
  };
  const el = badge(state.label, state.tone, state.icon, tips[state.id]);
  el.setAttribute('data-testid', 'model-status');
  el.dataset.state = state.id;
  return el;
}

/**
 * Packages that are missing for a model, including those of the models it builds on:
 * [{id, missing: [module names], hint, error}].
 */
export function packageProblems(model) {
  const out = [];
  const seen = new Set();
  const visit = (id, status) => {
    if (!status || seen.has(id)) return;
    seen.add(id);
    const missing = Array.isArray(status.missing_requirements) ? status.missing_requirements : [];
    if (status.error) out.push({ id, missing: [], hint: '', error: String(status.error) });
    else if (missing.length) out.push({ id, missing, hint: status.install_hint || '', error: '' });
    for (const [depId, dep] of Object.entries(status.dependencies || {})) visit(depId, dep);
  };
  if (model) visit(model.id, model.status);
  return out;
}

/** The install hint of the API: a command (or sentence with a command) the user runs in a terminal. */
export function installHint(problem, modelId) {
  const what = problem.id === modelId ? 'This model' : `The model it builds on, “${problem.id}”,`;
  if (problem.error) {
    return h('div', { class: 'install-hint', testid: 'install-hint' }, h('div', { class: 'install-hint-text break' }, `${what} is not available: ${problem.error}`));
  }
  return h(
    'div',
    { class: 'install-hint', testid: 'install-hint' },
    h(
      'div',
      { class: 'install-hint-text break' },
      `${what} needs ${problem.missing.length === 1 ? 'a Python package that is' : 'Python packages that are'} not installed: `,
      h('span', { class: 'mono' }, problem.missing.join(', ')),
      '.',
    ),
    problem.hint
      ? [
          h('div', { class: 'install-hint-label' }, icon('terminal', 's'), 'Run in a terminal, in the Python environment of this application, then refresh:'),
          h('div', { class: 'install-hint-cmd' }, h('pre', { class: 'codeblock', testid: 'install-command' }, problem.hint), copyButton(problem.hint, 'Copy the hint')),
        ]
      : h('div', { class: 'install-hint-label' }, 'Install the package in the Python environment of this application, then refresh.'),
  );
}

// ------------------------------------------------------------------ capabilities

export function outputType(model) {
  const caps = (model && model.capabilities) || {};
  if (caps.alpha_type === 'alpha') {
    return { label: 'True alpha matte', tip: 'The network predicts a soft alpha matte itself: hair and motion blur keep their transparency.' };
  }
  if (caps.alpha_type === 'mask') {
    return { label: 'Mask refined to alpha', tip: 'The network predicts a segmentation mask; the matte refinement turns its edge into a soft alpha.' };
  }
  return { label: String(caps.alpha_type || 'unknown'), tip: '' };
}

export function subjectDomain(model) {
  const caps = (model && model.capabilities) || {};
  if (caps.subject_domain === 'human') return { label: 'People only', tip: 'Trained on people: other subjects are not matted reliably.' };
  if (caps.subject_domain === 'any') return { label: 'Any subject', tip: 'Class agnostic: people, animals, objects.' };
  return { label: String(caps.subject_domain || 'unknown'), tip: '' };
}

const PROMPT_NAMES = { points: 'clicks', box: 'box', mask: 'mask' };
const TEMPORAL = {
  recurrent: 'Recurrent (state carried from frame to frame)',
  memory: 'Memory (looks back at earlier frames)',
  none: 'None (every frame on its own)',
};

function fact(label, value, tip = null) {
  return h('div', { class: 'mfact', tip }, h('span', { class: 'mfact-label' }, label), h('span', { class: 'mfact-value' }, value));
}

/** The facts a user needs to choose a matting model: what it outputs and what it can follow. */
export function capabilityFacts(model, { extended = false } = {}) {
  const caps = model.capabilities || {};
  const output = outputType(model);
  const domain = subjectDomain(model);
  const prompts = (caps.native_prompts || []).map((name) => PROMPT_NAMES[name] || name);
  const facts = [
    fact('Output', output.label, output.tip),
    fact('Subject', domain.label, domain.tip),
    fact(
      'Subject selection',
      caps.tracks_single_subject ? 'Follows the selected subject' : 'None in the model',
      caps.tracks_single_subject
        ? 'The model follows one selected subject and ignores other people and objects.'
        : 'The model mattes everything it recognises. A selection is applied afterwards by connected-component tracking, which cannot separate subjects that touch.',
    ),
  ];
  if (extended) {
    facts.push(
      fact('Understands', prompts.length ? prompts.join(', ') : 'no prompts', 'Kinds of selection the model takes directly.'),
      fact('Temporal consistency', TEMPORAL[caps.temporal] || String(caps.temporal || 'unknown')),
      fact('Batch size', caps.max_batch_size ? `at most ${caps.max_batch_size}` : 'no limit'),
      fact('Half precision', caps.supports_fp16 ? 'supported (fp16 on CUDA)' : 'not supported'),
      fact('Devices', (caps.devices || []).map((d) => d.toUpperCase()).join(', ') || 'unknown'),
    );
    if (caps.needs_initial_mask) facts.push(fact('Start', 'Needs a mask of the subject', 'A helper model turns your selection (or the automatic detection) into a mask on the reference frame.'));
    if (caps.supports_reverse === false) facts.push(fact('Backward tracking', 'not supported', 'Frames before the reference frame of the subject cannot be matted.'));
    if (caps.two_pass) facts.push(fact('Order of work', 'Analyses the whole clip first', 'The first frame is written only after every frame was analysed; an interrupted job starts again.'));
  }
  return h('div', { class: 'mfacts', testid: 'model-facts' }, facts);
}

// ------------------------------------------------------------------ licence

const COMMERCIAL = {
  allowed: { label: 'Commercial use allowed', tone: 'ok', icon: 'check' },
  conditional: { label: 'Commercial use: conditions apply', tone: 'warn', icon: 'alert' },
  prohibited: { label: 'Commercial use prohibited', tone: 'err', icon: 'x' },
};

export function commercialBadge(license) {
  const value = license ? license.commercial_use : null;
  const info = COMMERCIAL[value] || { label: `Commercial use: ${value || 'unknown'}`, tone: 'warn', icon: 'help' };
  const el = badge(info.label, info.tone, info.icon);
  el.setAttribute('data-testid', 'license-badge');
  el.dataset.commercial = String(value);
  return el;
}

/** Licence name (linked to its text when the API gives a URL), the commercial-use badge and the notes. */
export function licenseBlock(model, { notes = true } = {}) {
  const license = model.license || {};
  const name = license.name || 'Unknown licence';
  const isWebUrl = typeof license.url === 'string' && /^https?:\/\//i.test(license.url);
  return h(
    'div',
    { class: 'mlicense', testid: 'model-license' },
    h(
      'div',
      { class: 'mlicense-head' },
      icon('shield', 's'),
      h('span', { class: 'mlicense-label' }, 'Licence'),
      isWebUrl
        ? h('a', { class: 'mlicense-name', href: license.url, target: '_blank', rel: 'noopener noreferrer', tip: 'Open the licence text (needs an internet connection)' }, name, icon('external', 's'))
        : h('span', { class: 'mlicense-name' }, name),
      commercialBadge(license),
    ),
    notes && license.notes ? h('p', { class: 'mlicense-notes break' }, license.notes) : null,
  );
}

// ------------------------------------------------------------------ limitations, variants

export function limitationsDisclosure(model, { open = false } = {}) {
  const items = model.limitations || [];
  if (!items.length) return null;
  return disclosure(`Known limitations (${items.length})`, bullets(items), { open, plain: true, testid: 'model-limitations' });
}

/** "about 1.2 GB VRAM at 1080p" or null when the model gives no figure. */
export function vramHint(variant) {
  return variant && variant.vram_hint_mb ? `about ${formatMb(variant.vram_hint_mb)} VRAM at 1080p` : null;
}

/** Sum of the sizes the API states for the weight files of a variant (null when none is stated). */
export function variantBytes(variant) {
  const sizes = (variant.weights || []).map((weight) => weight.size_bytes).filter((size) => typeof size === 'number');
  return sizes.length ? sizes.reduce((a, b) => a + b, 0) : null;
}
