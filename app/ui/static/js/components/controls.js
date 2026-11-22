// Building blocks shared by all views: buttons, badges, notes, empty states, form controls.
// Controls return {el, set(...)} so callers can keep them in sync with their model without
// re-creating DOM nodes (which would steal focus while typing or dragging).

import { h, fill, clamp, setPercent, uid } from '../dom.js';
import { icon } from '../icons.js';
import { toApiError } from '../api.js';
import { toast } from './toast.js';

// ------------------------------------------------------------------ buttons, badges

export function button({ label, icon: iconName, iconRight, variant = '', size = '', onClick, testid, tip, disabled = false, ariaLabel, type = 'button' }) {
  return h(
    'button',
    {
      class: ['btn', variant, size, !label && 'icon-only'],
      type,
      onClick,
      testid,
      tip,
      disabled,
      'aria-label': ariaLabel || (!label ? tip : null),
    },
    iconName ? icon(iconName, size === 'small' ? 's' : '') : null,
    label || null,
    iconRight ? icon(iconRight, size === 'small' ? 's' : '') : null,
  );
}

export function linkButton({ label, icon: iconName, href, variant = '', size = '', testid, tip, download = null, newTab = false }) {
  return h(
    'a',
    {
      class: ['btn', variant, size],
      href,
      testid,
      tip,
      download: download === null ? null : download,
      target: newTab ? '_blank' : null,
      rel: newTab ? 'noopener noreferrer' : null,
    },
    iconName ? icon(iconName, size === 'small' ? 's' : '') : null,
    label,
  );
}

/** Run an async click handler while showing a spinner in the button and blocking repeats. */
export function busyWhile(buttonEl, task) {
  if (buttonEl.classList.contains('is-busy')) return Promise.resolve();
  buttonEl.classList.add('is-busy');
  buttonEl.setAttribute('aria-busy', 'true');
  const spinner = h('span', { class: 'spinner' });
  const first = buttonEl.firstElementChild;
  const hadIcon = first && first.classList.contains('icon');
  if (hadIcon) first.replaceWith(spinner);
  else buttonEl.prepend(spinner);
  return Promise.resolve()
    .then(task)
    .finally(() => {
      buttonEl.classList.remove('is-busy');
      buttonEl.removeAttribute('aria-busy');
      if (hadIcon) spinner.replaceWith(first);
      else spinner.remove();
    });
}

export function badge(text, tone = '', iconName = null, tip = null) {
  return h('span', { class: ['badge', tone], tip }, iconName ? icon(iconName) : null, text);
}

// ------------------------------------------------------------------ notes, errors, empty states

const NOTE_ICON = { info: 'info', ok: 'check-circle', warn: 'alert', err: 'x-circle' };

export function note({ tone = '', title = '', text = '', hint = '', actions = null, iconName = null, testid = null, children = null }) {
  return h(
    'div',
    { class: ['note', tone], testid, role: tone === 'err' ? 'alert' : null },
    icon(iconName || NOTE_ICON[tone] || 'info'),
    h(
      'div',
      { class: 'note-body' },
      title ? h('div', { class: 'note-title break' }, title) : null,
      text ? h('div', { class: 'break' }, text) : null,
      hint ? h('div', { class: 'note-hint break' }, hint) : null,
      children,
      actions && actions.length ? h('div', { class: 'note-actions' }, actions) : null,
    ),
  );
}

/** A note that shows an API error: its message, the hint and (optionally) follow-up actions. */
export function errorNote(error, { title = '', actions = null, testid = 'error-note', tone = 'err' } = {}) {
  const err = toApiError(error);
  return note({
    tone,
    title: title || err.message,
    text: title ? err.message : '',
    hint: err.hint || '',
    actions,
    testid,
  });
}

export function emptyState({ iconName = 'info', title, text = '', actions = null, compact = false, plain = false, testid = null }) {
  return h(
    'div',
    { class: ['empty', compact && 'compact', plain && 'plain'], testid },
    icon(iconName, 'xl'),
    h('div', { class: 'empty-title' }, title),
    text ? h('div', { class: 'empty-text' }, text) : null,
    actions && actions.length ? h('div', { class: 'btn-row' }, actions) : null,
  );
}

export function loadingBlock(label = 'Loading') {
  return h('div', { class: 'empty plain compact', role: 'status' }, h('span', { class: 'spinner large' }), h('div', { class: 'muted' }, label));
}

export function skeletonCards(count = 3) {
  return h(
    'div',
    { class: 'stack', 'aria-hidden': 'true' },
    Array.from({ length: count }, () => h('div', { class: 'skeleton skeleton-card' })),
  );
}

// ------------------------------------------------------------------ progress

export function progressBar({ large = false, tone = '', testid = null, label = 'Progress' } = {}) {
  const barFill = h('span');
  const el = h('div', { class: ['progress', large && 'large', tone], role: 'progressbar', 'aria-label': label, 'aria-valuemin': '0', 'aria-valuemax': '100', testid }, barFill);
  return {
    el,
    /** percent 0..100, or null for "in progress, amount unknown". */
    set(percent) {
      if (percent === null || percent === undefined || !Number.isFinite(percent)) {
        el.classList.add('indeterminate');
        el.removeAttribute('aria-valuenow');
      } else {
        el.classList.remove('indeterminate');
        setPercent(el, percent);
        el.setAttribute('aria-valuenow', String(Math.round(percent)));
      }
    },
    setTone(next) {
      el.classList.remove('ok', 'warn', 'err');
      if (next) el.classList.add(next);
    },
  };
}

// ------------------------------------------------------------------ layout helpers

export function card({ title = null, sub = null, actions = null, body, flush = false, testid = null, foot = null, cls = '' }) {
  return h(
    'section',
    { class: ['card', cls], testid },
    title || actions
      ? h('header', { class: 'card-head' }, h('h2', null, title, sub ? h('span', { class: 'card-sub' }, sub) : null), actions)
      : null,
    h('div', { class: ['card-body', flush && 'flush'] }, body),
    foot ? h('footer', { class: 'card-foot' }, foot) : null,
  );
}

/** <dl> from [label, value] pairs; pairs with a null / undefined value are skipped. */
export function kv(pairs, testid = null) {
  const rows = [];
  for (const [label, value] of pairs) {
    if (value === null || value === undefined || value === '') continue;
    rows.push(h('dt', null, label), h('dd', null, value));
  }
  return h('dl', { class: 'kv', testid }, rows);
}

export function disclosure(summary, body, { open = false, plain = false, testid = null } = {}) {
  return h(
    'details',
    { class: ['disclosure', plain && 'plain'], open, testid },
    h('summary', null, icon('chevron-right', 's chev'), summary),
    h('div', { class: 'disclosure-body' }, body),
  );
}

export function bullets(items, tone = '') {
  return h(
    'ul',
    { class: ['bullets', tone] },
    items.map((item) => h('li', null, item)),
  );
}

export function copyButton(text, label = 'Copy') {
  return button({
    icon: 'copy',
    size: 'small',
    variant: 'ghost',
    tip: label,
    onClick: async () => {
      try {
        await navigator.clipboard.writeText(text);
        toast({ type: 'ok', title: 'Copied to the clipboard', timeout: 2000 });
      } catch {
        toast({ type: 'warn', title: 'Copying is not available here', text: 'Select the text and copy it manually.' });
      }
    },
  });
}

// ------------------------------------------------------------------ segmented control

export function segmented({ options, value, onChange, label, testid = null, block = false }) {
  const el = h('div', { class: ['segmented', block && 'block'], role: 'radiogroup', 'aria-label': label, testid });
  let current = value;
  let items = [];

  function render(list) {
    items = list;
    fill(
      el,
      list.map((option) =>
        h(
          'button',
          {
            type: 'button',
            role: 'radio',
            'aria-checked': String(option.value === current),
            disabled: !!option.disabled,
            tip: option.tip || null,
            dataset: { value: String(option.value) },
            testid: testid ? `${testid}-${option.value}` : null,
            onClick: () => {
              if (option.value === current) return;
              current = option.value;
              sync();
              onChange(option.value);
            },
          },
          option.icon ? icon(option.icon, 's') : null,
          option.label,
        ),
      ),
    );
  }

  function sync() {
    Array.from(el.children).forEach((child, index) => {
      child.setAttribute('aria-checked', String(items[index].value === current));
    });
  }

  el.addEventListener('keydown', (event) => {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
    const enabled = items.filter((item) => !item.disabled);
    const position = enabled.findIndex((item) => item.value === current);
    if (position < 0) return;
    const step = event.key === 'ArrowLeft' || event.key === 'ArrowUp' ? -1 : 1;
    const next = enabled[(position + step + enabled.length) % enabled.length];
    event.preventDefault();
    event.stopPropagation();
    current = next.value;
    sync();
    const target = Array.from(el.children)[items.indexOf(next)];
    if (target) target.focus();
    onChange(next.value);
  });

  render(options);
  return {
    el,
    set(next) {
      current = next;
      sync();
    },
    setOptions(list) {
      render(list);
    },
  };
}

// ------------------------------------------------------------------ select

/**
 * Native select. Options: [{value, label, disabled, group}] (value may be null, a number ...).
 * `setOptions` only rebuilds the list when it really changed.
 */
export function select({ options, value, onChange, label = null, testid = null, id = null }) {
  const el = h('select', { class: 'select', 'aria-label': label, testid, id });
  let flat = [];
  let signature = '';

  function render(list, selected) {
    const sig = JSON.stringify(list);
    if (sig !== signature) {
      signature = sig;
      flat = list;
      const nodes = [];
      let group = null;
      let groupName = null;
      list.forEach((option, index) => {
        // data-value carries the API value so tests and styles can address an option by meaning
        const node = h('option', { value: String(index), disabled: !!option.disabled, dataset: { value: String(option.value) } }, option.label);
        if (option.group) {
          if (option.group !== groupName) {
            groupName = option.group;
            group = h('optgroup', { label: option.group });
            nodes.push(group);
          }
          group.appendChild(node);
        } else {
          groupName = null;
          nodes.push(node);
        }
      });
      fill(el, nodes);
    }
    const index = flat.findIndex((option) => option.value === selected);
    el.selectedIndex = -1;
    el.value = index >= 0 ? String(index) : '';
  }

  el.addEventListener('change', () => {
    const option = flat[Number(el.value)];
    if (option) onChange(option.value);
  });

  render(options, value);
  return {
    el,
    set(next) {
      const index = flat.findIndex((option) => option.value === next);
      el.value = index >= 0 ? String(index) : '';
    },
    setOptions(list, selected) {
      render(list, selected);
    },
  };
}

// ------------------------------------------------------------------ switch, checkbox

export function switchControl({ checked = false, onChange, label, description = '', testid = null, disabled = false }) {
  const input = h('input', { type: 'checkbox', role: 'switch', checked, disabled, testid, onChange: () => onChange(input.checked) });
  const el = h(
    'label',
    { class: 'switch' },
    input,
    h('span', { class: 'switch-track' }),
    h('span', { class: 'switch-text' }, h('span', null, label), description ? h('span', { class: 'switch-desc' }, description) : null),
  );
  return {
    el,
    input,
    set(next, { preset = false } = {}) {
      input.checked = !!next;
      el.classList.toggle('is-preset', preset);
    },
    setDisabled(value) {
      input.disabled = !!value;
    },
  };
}

export function checkbox({ checked = false, onChange, label, description = '', testid = null, disabled = false }) {
  const input = h('input', { type: 'checkbox', checked, disabled, testid, onChange: () => onChange(input.checked) });
  const el = h(
    'label',
    { class: 'check' },
    input,
    h('span', { class: 'check-text' }, h('span', null, label), description ? h('span', { class: 'check-desc' }, description) : null),
  );
  return {
    el,
    input,
    set(next) {
      input.checked = !!next;
    },
    setDisabled(value) {
      input.disabled = !!value;
    },
  };
}

// ------------------------------------------------------------------ slider + number box

/**
 * A slider paired with a number box. Values are API values; `scale` multiplies them for display
 * (0.0005 -> 0.05 %). `soft` = [min, max] range of the slider when it should be narrower than the
 * range the API accepts.
 */
export function sliderNumber({ min, max, step, soft = null, scale = 1, digits = 2, value = null, onChange, label, testid = null, noSlider = false }) {
  const lo = (soft ? soft[0] : min) * scale;
  const hi = (soft ? soft[1] : max) * scale;
  const shown = (apiValue) => (apiValue === null || apiValue === undefined ? '' : String(parseFloat((apiValue * scale).toFixed(Math.max(digits, 0) + 2))));
  const snap = (display) => {
    const apiValue = clamp(display / scale, min, max);
    return parseFloat(apiValue.toPrecision(10));
  };

  const range = noSlider
    ? null
    : h('input', { class: 'range', type: 'range', min: String(lo), max: String(hi), step: String(step * scale), 'aria-label': label, testid: testid ? `${testid}-range` : null });
  const number = h('input', {
    class: 'input num',
    type: 'number',
    min: String(min * scale),
    max: String(max * scale),
    step: String(step * scale),
    inputMode: 'decimal',
    'aria-label': label,
    testid: testid ? `${testid}-number` : null,
  });

  function paint(apiValue) {
    if (!range) return;
    const display = apiValue === null || apiValue === undefined ? lo : apiValue * scale;
    range.value = String(clamp(display, lo, hi));
    setPercent(range, hi > lo ? ((clamp(display, lo, hi) - lo) / (hi - lo)) * 100 : 0);
  }

  if (range) {
    range.addEventListener('input', () => {
      const apiValue = snap(Number(range.value));
      number.value = shown(apiValue);
      paint(apiValue);
      onChange(apiValue);
    });
  }
  number.addEventListener('input', () => {
    if (number.value.trim() === '') return;
    const parsed = Number(number.value);
    if (!Number.isFinite(parsed)) return;
    if (parsed < min * scale || parsed > max * scale) return; // corrected when the box loses focus
    const apiValue = snap(parsed);
    paint(apiValue);
    onChange(apiValue);
  });
  number.addEventListener('change', () => {
    if (number.value.trim() === '') {
      number.value = shown(lastValue);
      return;
    }
    const parsed = Number(number.value);
    if (!Number.isFinite(parsed)) {
      number.value = shown(lastValue);
      return;
    }
    const apiValue = snap(parsed);
    number.value = shown(apiValue);
    paint(apiValue);
    onChange(apiValue);
  });

  let lastValue = value;
  const el = h('div', { class: noSlider ? 'input-group' : 'slider-pair' }, range, number);
  const api = {
    el,
    set(apiValue, { preset = false } = {}) {
      lastValue = apiValue;
      if (document.activeElement !== number) number.value = shown(apiValue);
      paint(apiValue);
      if (range) range.classList.toggle('is-preset', preset);
    },
    setDisabled(disabled) {
      if (range) range.disabled = !!disabled;
      number.disabled = !!disabled;
    },
  };
  api.set(value);
  return api;
}

// ------------------------------------------------------------------ optional number / text

/**
 * A number box whose empty state means "automatic" (null). A value outside min..max is not
 * reported while typing; it is clamped when the box loses focus.
 */
export function optionalNumber({ min = 0, max = null, step = 1, integer = true, value = null, placeholder = 'Automatic', onChange, label, testid = null }) {
  const input = h('input', {
    class: 'input num',
    type: 'number',
    step: String(step),
    inputMode: integer ? 'numeric' : 'decimal',
    placeholder,
    'aria-label': label,
    testid,
  });
  let lo = min;
  let hi = max;
  let last = value === undefined ? null : value;

  const applyRange = () => {
    input.min = String(lo);
    if (hi === null) input.removeAttribute('max');
    else input.max = String(hi);
  };
  const valid = (number) => Number.isFinite(number) && number >= lo && (hi === null || number <= hi) && (!integer || Number.isInteger(number));
  const report = (next) => {
    if (next === last) return;
    last = next;
    onChange(next);
  };

  input.addEventListener('input', () => {
    const text = input.value.trim();
    if (text === '') {
      input.classList.remove('is-invalid');
      report(null);
      return;
    }
    const number = Number(text);
    input.classList.toggle('is-invalid', !valid(number));
    if (valid(number)) report(number);
  });
  input.addEventListener('change', () => {
    const text = input.value.trim();
    input.classList.remove('is-invalid');
    if (text === '') return;
    let number = Number(text);
    if (!Number.isFinite(number)) {
      input.value = last === null ? '' : String(last);
      return;
    }
    if (integer) number = Math.round(number);
    number = clamp(number, lo, hi === null ? Number.MAX_SAFE_INTEGER : hi);
    input.value = String(number);
    report(number);
  });

  applyRange();
  input.value = last === null ? '' : String(last);
  return {
    el: input,
    set(next) {
      last = next === undefined ? null : next;
      if (document.activeElement !== input) {
        input.value = last === null ? '' : String(last);
        input.classList.remove('is-invalid');
      }
    },
    setRange(nextMin, nextMax) {
      lo = nextMin;
      hi = nextMax;
      applyRange();
    },
    setPlaceholder(text) {
      input.placeholder = text;
    },
    setDisabled(disabled) {
      input.disabled = !!disabled;
    },
  };
}

/** A text box whose empty state means "automatic" (null). The trimmed text is reported as typed. */
export function optionalText({ value = null, placeholder = 'Automatic', onChange, label, testid = null, mono = false, maxLength = 200 }) {
  const input = h('input', {
    class: ['input', mono && 'mono'],
    type: 'text',
    placeholder,
    maxLength,
    spellcheck: false,
    autocomplete: 'off',
    'aria-label': label,
    testid,
  });
  let last = value || null;
  input.value = last || '';
  input.addEventListener('input', () => {
    const next = input.value.trim() || null;
    if (next === last) return;
    last = next;
    onChange(next);
  });
  return {
    el: input,
    set(next) {
      last = next || null;
      if (document.activeElement !== input) input.value = last || '';
    },
    setInvalid(invalid) {
      input.classList.toggle('is-invalid', !!invalid);
    },
    setPlaceholder(text) {
      input.placeholder = text;
    },
    setDisabled(disabled) {
      input.disabled = !!disabled;
    },
  };
}

// ------------------------------------------------------------------ field wrapper

/**
 * Label + control + hint + error slot. `path` is the settings path the field edits; server-side
 * validation messages for that path are shown under the control.
 */
export function field({ label, control, hint = '', path = null, labelFor = null, testid = null, wide = false }) {
  // labelFor: false = the label names a group of controls and must not activate the first of them
  const id = labelFor === false ? null : labelFor || uid('f');
  const tag = h('span', { class: 'field-tag', hidden: true });
  const reset = h('span', { class: 'field-reset' });
  const hintEl = h('div', { class: 'field-hint', hidden: !hint }, hint);
  const noteEl = h('div', { class: 'field-hint', hidden: true });
  const errorEl = h('div', { class: 'field-error', hidden: true, role: 'alert' });
  const el = h(
    'div',
    { class: ['field', wide && 'wide'], dataset: { field: path }, testid },
    h('div', { class: 'field-head' }, h('label', { class: 'field-label', for: id }, label), tag, reset),
    control,
    hintEl,
    noteEl,
    errorEl,
  );
  if (id) {
    const target = control.matches && control.matches('input, select, textarea') ? control : control.querySelector('input, select, textarea');
    if (target && !target.id) target.id = id;
  }
  return {
    el,
    setError(message) {
      el.classList.toggle('has-error', !!message);
      errorEl.hidden = !message;
      fill(errorEl, message ? [icon('alert', 's'), h('span', { class: 'break' }, message)] : []);
    },
    setTag(text, changed = false) {
      tag.hidden = !text;
      tag.textContent = text || '';
      tag.classList.toggle('is-changed', changed);
    },
    setReset(node) {
      fill(reset, node || []);
    },
    setHint(text) {
      hintEl.hidden = !text;
      hintEl.textContent = text || '';
    },
    /** A transient remark (for example "the container was reset"). */
    setNote(text) {
      noteEl.hidden = !text;
      noteEl.textContent = text || '';
    },
  };
}
