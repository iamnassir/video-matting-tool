// Models: every matting model and helper model of this installation with its licence, its
// installation state and - where weight files are missing - an explicit, user-triggered download.

import { h, fill, keyedList } from '../dom.js';
import { store, config, modelById, restrictedLicenses } from '../state.js';
import { reconcile } from '../config.js';
import { formatBytes, formatPercent, isNumber, plural } from '../format.js';
import { cancelDownload, dismissDownload, loadModels, navigate, startDownload } from '../actions.js';
import { badge, busyWhile, button, checkbox, disclosure, emptyState, errorNote, kv, note, progressBar, skeletonCards } from '../components/controls.js';
import {
  capabilityFacts,
  installHint,
  licenseBlock,
  limitationsDisclosure,
  modelState,
  packageProblems,
  statusBadge,
  variantBytes,
  vramHint,
} from '../components/modelinfo.js';

// ------------------------------------------------------------------ what a download would fetch

/**
 * What POST /api/models/{id}/download fetches for a variant: the missing files of the variant
 * itself plus the missing default-variant files of the models it builds on.
 */
function pendingDownload(state, model, variant) {
  const status = model.status || {};
  const own = (status.variants || {})[variant.id] || null;
  const missing = (entry) => (entry && Array.isArray(entry.files) ? entry.files.filter((file) => !file.present).length : 0);
  let bytes = own ? own.download_bytes || 0 : 0;
  let files = missing(own);
  const dependencies = [];
  for (const depId of model.depends_on || []) {
    const dep = modelById(state, depId);
    const depStatus = (status.dependencies || {})[depId];
    const depVariant = dep && depStatus && depStatus.variants ? depStatus.variants[dep.default_variant] : null;
    if (depVariant && !depVariant.ready) {
      bytes += depVariant.download_bytes || 0;
      files += missing(depVariant);
      dependencies.push(dep.name);
    }
  }
  return { ownReady: !!(own && own.ready), files, bytes, dependencies };
}

// ------------------------------------------------------------------ one variant

function fileList(model, variant) {
  const status = ((model.status || {}).variants || {})[variant.id] || {};
  const present = new Map((status.files || []).map((file) => [file.filename, file]));
  if (!(variant.weights || []).length) return h('p', { class: 'muted' }, 'This variant has no weight files of its own.');
  return h(
    'ul',
    { class: 'wfiles' },
    variant.weights.map((weight) => {
      const state = present.get(weight.filename) || {};
      const size = isNumber(state.expected_bytes) ? state.expected_bytes : weight.size_bytes;
      return h(
        'li',
        { class: 'wfile' },
        h('span', { class: ['status-dot', state.present ? 'ok' : 'warn'], tip: state.present ? 'On disk' : 'Not downloaded' }),
        h(
          'div',
          { class: 'wfile-main' },
          h('span', { class: 'mono break' }, weight.filename),
          h(
            'span',
            { class: 'wfile-meta break' },
            [isNumber(size) ? formatBytes(size) : null, weight.description || null, weight.source || null].filter(Boolean).join(' · '),
          ),
          h('span', { class: 'wfile-url mono break' }, state.url || weight.url || ''),
        ),
      );
    }),
  );
}

function variantRow(model, variant, context) {
  const key = `${model.id}/${variant.id}`;
  const stateSlot = h('div', { class: 'vrow-state' });
  const detailSlot = h('div', { class: 'vrow-detail' });
  const bar = progressBar({ testid: 'download-progress', label: `Download of ${model.name}, ${variant.label}` });
  const progressText = h('span', { class: 'num faint', testid: 'download-status' });
  let startError = null;
  let shown = '';

  const presets = Object.entries(model.quality_variants || {})
    .filter(([, id]) => id === variant.id)
    .map(([quality]) => quality);
  const size = variantBytes(variant);
  const meta = [variant.description || null, vramHint(variant), isNumber(size) ? formatBytes(size) : null].filter(Boolean).join(' · ');

  const begin = (buttonEl) => {
    startError = null;
    busyWhile(buttonEl, async () => {
      try {
        await startDownload(model.id, variant.id, context.accepted());
      } catch (error) {
        startError = error;
        shown = '';
        render();
      }
    });
  };

  function render() {
    const state = store.get();
    const task = state.downloads[key] || null;
    const pending = pendingDownload(state, model, variant);
    const blocked = context.needsAcceptance() && !context.accepted();
    const signature = JSON.stringify([task ? [task.status, task.error] : null, pending, blocked, !!startError]);
    if (task && task.status === 'running') {
      // progress changes several times per second: only the bar and the text are touched
      const percent = task.total > 0 ? (task.downloaded / task.total) * 100 : null;
      bar.set(percent);
      progressText.textContent = [
        task.total > 0 ? `${formatBytes(task.downloaded)} of ${formatBytes(task.total)}` : formatBytes(task.downloaded),
        percent !== null ? formatPercent(percent) : null,
        task.file || null,
      ]
        .filter(Boolean)
        .join(' · ');
    }
    if (signature === shown) return;
    shown = signature;
    el.dataset.ready = String(pending.ownReady && pending.files === 0);

    if (task && task.status === 'running') {
      fill(stateSlot, button({ label: 'Cancel', icon: 'x', size: 'small', testid: 'download-cancel', onClick: () => cancelDownload(key) }));
      fill(detailSlot, h('div', { class: 'vrow-progress' }, bar.el, progressText));
      return;
    }
    const downloadButton = () => {
      const label = pending.bytes > 0 ? `Download ${formatBytes(pending.bytes)}` : 'Download';
      const buttonEl = button({
        label,
        icon: 'download',
        size: 'small',
        variant: 'primary',
        testid: 'download-start',
        disabled: blocked,
        tip: blocked ? 'Tick the licence acknowledgement of this model first' : `Download ${plural(pending.files, 'file')} into the models folder`,
      });
      buttonEl.addEventListener('click', () => begin(buttonEl));
      return buttonEl;
    };
    const details = [];
    if (task && task.status === 'failed') {
      details.push(
        errorNote(task.error || { message: 'The download failed.' }, {
          title: 'The download failed',
          testid: 'download-error',
          actions: [button({ label: 'Dismiss', size: 'small', testid: 'download-dismiss', onClick: () => dismissDownload(key) })],
        }),
      );
    } else if (task && task.status === 'cancelled') {
      details.push(
        note({
          tone: 'info',
          text: 'The download was cancelled. Files that were complete are kept.',
          testid: 'download-cancelled',
          actions: [button({ label: 'Dismiss', size: 'small', testid: 'download-dismiss', onClick: () => dismissDownload(key) })],
        }),
      );
    }
    if (startError) details.push(errorNote(startError, { title: 'The download could not be started', testid: 'download-error' }));
    if (pending.files > 0) {
      fill(stateSlot, downloadButton());
      if (pending.dependencies.length) {
        details.unshift(h('div', { class: 'field-hint' }, `Includes the weights of ${pending.dependencies.join(' and ')}, which this model builds on.`));
      }
    } else if (pending.ownReady) {
      fill(stateSlot, badge('On disk', 'ok', 'check'));
    } else {
      // nothing the API could download: the size / location of the files is unknown to it
      fill(stateSlot, badge('Not available', 'warn', 'alert', 'The server reports this variant as not ready but lists no file to download'));
    }
    fill(detailSlot, details);
  }

  const el = h(
    'li',
    { class: 'vrow', testid: 'model-variant', dataset: { variantId: variant.id } },
    h(
      'div',
      { class: 'vrow-head' },
      h(
        'div',
        { class: 'vrow-main' },
        h(
          'div',
          { class: 'vrow-title' },
          h('span', { class: 'vrow-label' }, variant.label || variant.id),
          variant.id === model.default_variant ? badge('Default', 'plain', null, 'Used when no variant and no quality preset decides otherwise') : null,
          presets.length ? h('span', { class: 'vrow-presets faint', tip: 'Quality presets that use this variant' }, presets.join(', ')) : null,
        ),
        meta ? h('div', { class: 'vrow-meta break' }, meta) : null,
      ),
      stateSlot,
    ),
    detailSlot,
    (variant.weights || []).length ? disclosure(plural(variant.weights.length, 'file'), fileList(model, variant), { plain: true, testid: 'variant-files' }) : null,
  );
  render();
  return { el, render };
}

// ------------------------------------------------------------------ one model

function modelCard(model, context) {
  const state = store.get();
  const stateInfo = modelState(model);
  const isMatting = model.kind !== 'auxiliary';
  const selected = isMatting && config.get('model') === model.id;
  const restricted = restrictedLicenses(state, model);
  const anyPending = (model.variants || []).some((variant) => pendingDownload(state, model, variant).files > 0);
  const rows = [];

  const rowContext = {
    needsAcceptance: () => restricted.length > 0,
    accepted: () => context.accepted.get(model.id) === true,
  };
  for (const variant of model.variants || []) rows.push(variantRow(model, variant, rowContext));

  const acknowledgement =
    restricted.length && anyPending
      ? checkbox({
          checked: rowContext.accepted(),
          label: 'I have read the licence terms and accept them',
          description: restricted.map((item) => `${item.name}: ${item.license.name}`).join(' · '),
          testid: 'license-accept',
          onChange: (value) => {
            context.accepted.set(model.id, value);
            rows.forEach((row) => row.render());
          },
        })
      : null;

  const problems = packageProblems(model);
  const dependencies = (model.depends_on || []).map((id) => {
    const dep = modelById(state, id);
    return dep ? `${dep.name} (${modelState(dep).label.toLowerCase()})` : `${id} (not registered)`;
  });

  const select = button({
    label: 'Use for new jobs',
    size: 'small',
    testid: 'model-use',
    tip: 'Select this model in Settings',
    onClick: () => {
      config.set('model', model.id);
      reconcile(config, store.get().options, 'model');
    },
  });

  const el = h(
    'li',
    { class: ['mcard', selected && 'is-selected'], testid: 'model-card', dataset: { modelId: model.id, state: stateInfo.id, kind: model.kind || 'matting' } },
    h(
      'div',
      { class: 'mcard-head' },
      h('div', { class: 'mcard-title' }, h('h3', { testid: 'model-name' }, model.name), h('span', { class: 'mcard-id mono' }, model.id)),
      h('div', { class: 'badge-row' }, statusBadge(model), selected ? badge('Selected in Settings', 'accent', 'check') : null),
      isMatting && !selected && stateInfo.id !== 'package' ? select : null,
    ),
    model.description ? h('p', { class: 'mcard-desc break' }, model.description) : null,
    isMatting ? capabilityFacts(model, { extended: true }) : null,
    licenseBlock(model),
    problems.length
      ? note({
          tone: 'err',
          iconName: 'terminal',
          title: 'Not installed',
          children: problems.map((problem) => installHint(problem, model.id)),
          testid: 'model-not-installed',
        })
      : null,
    dependencies.length ? h('div', { class: 'mcard-deps' }, h('span', { class: 'faint' }, 'Builds on: '), dependencies.join(', ')) : null,
    h('div', { class: 'section-title' }, (model.variants || []).length === 1 ? 'Weights' : 'Variants and weights'),
    h('ul', { class: 'vrows' }, rows.map((row) => row.el)),
    acknowledgement ? h('div', { class: 'mcard-accept' }, acknowledgement.el) : null,
    limitationsDisclosure(model),
    model.homepage || model.citation || (model.requires || []).length
      ? disclosure(
          'Project and citation',
          kv([
            ['Project page', /^https?:\/\//i.test(model.homepage || '') ? h('a', { href: model.homepage, target: '_blank', rel: 'noopener noreferrer' }, model.homepage) : model.homepage || null],
            ['Python packages', (model.requires || []).join(', ') || null],
            ['Citation', model.citation || null],
          ]),
          { plain: true, testid: 'model-about' },
        )
      : null,
  );
  return { el, render: () => rows.forEach((row) => row.render()) };
}

// ------------------------------------------------------------------ view

function modelSection({ title, text, testid }) {
  const count = h('span', { class: 'card-sub' });
  const list = h('ul', { class: 'mlist', testid: `${testid}-list` });
  const el = h('section', { class: 'msection', testid, hidden: true }, h('div', { class: 'msection-head' }, h('h2', null, title, count), h('p', null, text)), list);
  return { el, list, count, update: keyedList(list) };
}

export function modelsView(scope) {
  // licence acknowledgements are deliberate, per-session acts: they are not remembered
  const context = { accepted: new Map() };
  let cards = new Map();

  const refresh = button({ label: 'Refresh', icon: 'refresh', testid: 'models-refresh', tip: 'Check again which packages and weight files are present' });
  refresh.addEventListener('click', () => busyWhile(refresh, () => loadModels()));

  const matting = modelSection({
    title: 'Matting models',
    text: 'Produce the matte. Choose one of them in Settings.',
    testid: 'models-matting',
  });
  const auxiliary = modelSection({
    title: 'Helper models',
    text: 'Used for subject selection: detecting the main subject, turning clicks and boxes into a mask, finding a subject from a text description. They are loaded only when a job or the subject preview needs them.',
    testid: 'models-auxiliary',
  });
  const stateSlot = h('div');

  scope.add(
    store.watch(
      (state) => [state.models, state.modelsError, state.configVersion, state.options],
      () => {
        const state = store.get();
        const models = state.models;
        if (!models) {
          matting.el.hidden = true;
          auxiliary.el.hidden = true;
          fill(
            stateSlot,
            state.modelsError
              ? errorNote(state.modelsError, {
                  title: 'The model list could not be loaded',
                  testid: 'models-error',
                  actions: [button({ label: 'Retry', icon: 'refresh', size: 'small', onClick: () => loadModels() })],
                })
              : skeletonCards(3),
          );
          return;
        }
        if (models.length === 0) {
          matting.el.hidden = true;
          auxiliary.el.hidden = true;
          fill(stateSlot, emptyState({ iconName: 'package', title: 'No model is registered', text: 'This installation contains no model adapter. Check the server log for adapters that failed to load.', testid: 'models-empty' }));
          return;
        }
        fill(stateSlot);
        const next = new Map();
        const selected = config.get('model');
        const renderInto = (section, list) => {
          section.el.hidden = list.length === 0;
          const ready = list.filter((model) => model.status && model.status.ready).length;
          section.count.textContent = `${ready} of ${list.length} ready`;
          section.update(
            list,
            (model) => model.id,
            // the whole list is part of the signature: a card also shows the state of the models it builds on
            (model) => JSON.stringify([model, selected === model.id, (model.depends_on || []).map((id) => modelById(state, id))]),
            (model) => {
              const card = modelCard(model, context);
              next.set(model.id, card);
              return card.el;
            },
          );
          for (const model of list) if (!next.has(model.id) && cards.has(model.id)) next.set(model.id, cards.get(model.id));
        };
        renderInto(
          matting,
          models.filter((model) => model.kind !== 'auxiliary'),
        );
        renderInto(
          auxiliary,
          models.filter((model) => model.kind === 'auxiliary'),
        );
        cards = next;
      },
    ),
  );

  // download progress arrives several times per second: only the affected rows are updated
  scope.add(
    store.watch(
      (state) => state.downloads,
      () => cards.forEach((card) => card.render()),
      { immediate: false },
    ),
  );

  return h(
    'div',
    { class: 'models', testid: 'view-models' },
    h(
      'div',
      { class: 'view-head' },
      h(
        'div',
        { class: 'view-title' },
        h('h1', null, 'Models'),
        h('p', null, 'Models run on this computer. Weight files are downloaded only when you ask for it here; every model lists its licence and what that means for commercial work.'),
      ),
      h('div', { class: 'view-actions' }, button({ label: 'Settings', icon: 'sliders', onClick: () => navigate('settings'), testid: 'models-to-settings' }), refresh),
    ),
    stateSlot,
    h('div', { class: 'stack loose' }, matting.el, auxiliary.el),
  );
}
