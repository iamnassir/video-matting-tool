// Settings: how new jobs are processed. The form edits the shared configuration model; the panel
// beside it lists what differs from the defaults - which is exactly what is sent to the server.

import { h, fill } from '../dom.js';
import { store, config, automaticVariant, currentModel } from '../state.js';
import { EXPORTS, REFINEMENT_PRESET_LABELS, describeSettingValue, effectiveRefinementPreset, settingLabel } from '../config.js';
import { clearStartErrors, loadOptions, loadSystem, navigate } from '../actions.js';
import { plural } from '../format.js';
import { bullets, button, card, disclosure, kv, note } from '../components/controls.js';
import { createSettingsForm } from '../components/settingsform.js';

function backgroundValue(state) {
  const id = config.get('background');
  const entry = state.options ? state.options.backgrounds.find((item) => item.id === id) : null;
  const color = id === 'custom' ? config.get('background_color') : entry ? entry.color : null;
  return h(
    'span',
    { class: 'summary-swatch' },
    h('span', { class: ['swatch', id === 'transparent' && 'checker'], style: color ? { '--c': color } : null }),
    entry ? entry.label : String(id),
    color ? h('span', { class: 'mono faint' }, color) : null,
  );
}

/** The settings in effect, in words. Automatic values are named as such, never guessed. */
function effectiveRows(state) {
  const options = state.options;
  const model = currentModel(state);
  const explicitVariant = config.get('variant');
  const variant = model ? (explicitVariant ? model.variants.find((item) => item.id === explicitVariant) : automaticVariant(model, config.get('quality'))) : null;
  const refinement = effectiveRefinementPreset(config, options);
  const extras = EXPORTS.filter((item) => config.get(`exports.${item.key}`) === true).map((item) => item.label);
  const overrides = config.changedPaths().filter((path) => path.startsWith('refinement.') && path !== 'refinement.preset').length;
  return [
    ['Model', model ? model.name : String(config.get('model'))],
    ['Variant', variant ? `${variant.label}${explicitVariant ? '' : ' (automatic)'}` : explicitVariant],
    ['Quality', describeSettingValue('quality', config.get('quality'), options)],
    ['Device', config.get('device') === 'auto' ? 'automatic' : config.get('device')],
    ['Background', backgroundValue(state)],
    ['Refinement', `${REFINEMENT_PRESET_LABELS[refinement] || refinement || 'automatic'}${overrides ? ` + ${plural(overrides, 'change')}` : ''}`],
    ['Codec', describeSettingValue('encoding.codec', config.get('encoding.codec'), options)],
    ['Container', describeSettingValue('encoding.container', config.get('encoding.container'), options)],
    ['Resolution', describeSettingValue('encoding.resolution', config.get('encoding.resolution'), options)],
    ['Audio', config.get('encoding.preserve_audio') === false ? 'removed' : config.get('encoding.audio_codec') === 'auto' ? 'kept (copied when possible)' : `kept (${config.get('encoding.audio_codec')})`],
    ['Extra files', extras.length ? extras.join(', ') : 'none'],
    ['Frame limit', config.get('max_frames') === null ? null : `first ${config.get('max_frames')} frames`],
  ];
}

function summaryPanel(scope) {
  const effective = h('div');
  const changes = h('ul', { class: 'change-list', testid: 'settings-changes' });
  const changesTitle = h('div', { class: 'section-title' });
  const payload = h('pre', { class: 'codeblock', testid: 'settings-payload' });
  const resetAll = button({ label: 'Reset all', icon: 'undo', size: 'small', variant: 'ghost', testid: 'settings-reset-all', onClick: () => config.resetAll() });

  scope.add(
    store.watch(
      (state) => [state.configVersion, state.options, state.models],
      () => {
        const state = store.get();
        if (!state.options || !config.ready) {
          fill(effective, h('p', { class: 'muted' }, 'Loading…'));
          fill(changes);
          changesTitle.textContent = 'Changed from the defaults';
          resetAll.hidden = true;
          payload.textContent = '';
          return;
        }
        fill(effective, kv(effectiveRows(state), 'settings-effective'));
        const paths = config.changedPaths();
        changesTitle.textContent = paths.length ? `Changed from the defaults (${paths.length})` : 'Changed from the defaults';
        resetAll.hidden = paths.length === 0;
        fill(
          changes,
          paths.length
            ? paths.map((path) =>
                h(
                  'li',
                  { class: 'change-item', dataset: { path } },
                  h('span', { class: 'change-label' }, settingLabel(path)),
                  h('span', { class: 'change-value break' }, describeSettingValue(path, config.get(path), state.options)),
                  button({ icon: 'x', size: 'small', variant: 'ghost', tip: 'Back to the default', ariaLabel: `Reset: ${settingLabel(path)}`, onClick: () => config.reset(path) }),
                ),
              )
            : h('li', { class: 'change-none muted' }, 'Nothing: every setting is on its default.'),
        );
        payload.textContent = JSON.stringify(config.payload(), null, 2);
      },
    ),
  );

  return card({
    title: 'Summary',
    actions: resetAll,
    body: h(
      'div',
      { class: 'stack' },
      effective,
      h('div', null, changesTitle, changes),
      disclosure(
        'Sent with a new job',
        [h('p', { class: 'field-hint' }, 'Only settings you changed are sent; everything else stays with the server defaults, and “automatic” is sent as null.'), payload],
        { plain: true, testid: 'settings-payload-box' },
      ),
    ),
    testid: 'settings-summary',
    cls: 'settings-summary',
  });
}

export function settingsView(scope) {
  const form = createSettingsForm({ model: config, scope });
  const errorSlot = h('div', { testid: 'settings-errors', hidden: true });

  // validation messages of the last rejected start: next to their controls, the rest in a banner
  scope.add(
    store.watch(
      (state) => [state.startErrors, state.options],
      ([startErrors]) => {
        const rejected = startErrors && (Object.keys(startErrors.fields || {}).length || (startErrors.general || []).length);
        errorSlot.hidden = !rejected;
        if (!rejected) {
          form.setErrors({});
          fill(errorSlot);
          return;
        }
        const unplaced = form.setErrors(startErrors.fields);
        const lines = [...(startErrors.general || []), ...unplaced.map((item) => `${settingLabel(item.path)}: ${item.text}`)];
        const placed = Object.keys(startErrors.fields || {}).length - unplaced.length;
        fill(
          errorSlot,
          note({
            tone: 'err',
            title: 'The server did not accept these settings',
            text: placed ? `${plural(placed, 'setting')} below ${placed === 1 ? 'is' : 'are'} marked with the message of the server.` : '',
            children: lines.length ? bullets(lines) : null,
            actions: placed ? [button({ label: 'Show', size: 'small', onClick: () => form.revealFirstError() })] : null,
            testid: 'settings-rejected',
          }),
        );
      },
    ),
  );
  // a message about a setting is out of date as soon as the settings change
  scope.add(config.subscribe(() => clearStartErrors()));

  // device memory figures in the device list are those of the moment the view is opened
  loadSystem();
  if (!store.get().options) loadOptions();

  return h(
    'div',
    { class: 'settings', testid: 'view-settings' },
    h(
      'div',
      { class: 'view-head' },
      h(
        'div',
        { class: 'view-title' },
        h('h1', null, 'Settings'),
        h('p', null, 'Model, background, matte refinement and output of new jobs. Everything left on automatic follows the source video and the quality preset.'),
      ),
      h(
        'div',
        { class: 'view-actions' },
        button({ label: 'Subject', icon: 'chevron-left', onClick: () => navigate('subject'), testid: 'settings-back' }),
        button({ label: 'Next: process', iconRight: 'arrow-right', variant: 'primary', onClick: () => navigate('process'), testid: 'settings-next' }),
      ),
    ),
    h('div', { class: 'settings-layout' }, h('div', { class: 'stack' }, errorSlot, form.el), h('aside', { class: 'settings-side' }, summaryPanel(scope))),
  );
}
