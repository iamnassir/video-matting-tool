// Transient notifications in the lower right corner.

import { h } from '../dom.js';
import { icon } from '../icons.js';
import { toApiError } from '../api.js';

const ICON = { info: 'info', ok: 'check-circle', warn: 'alert', err: 'x-circle' };
const TIMEOUT = { info: 5000, ok: 4500, warn: 8000, err: 11000 };

let host = null;

function ensureHost() {
  if (!host) {
    host = h('div', { class: 'toasts', role: 'region', 'aria-label': 'Notifications', 'aria-live': 'polite', testid: 'toasts' });
    document.body.appendChild(host);
  }
  return host;
}

/**
 * Show a toast. `type`: info | ok | warn | err. `action`: {label, onClick} adds a button.
 * Returns a function that dismisses it.
 */
export function toast({ type = 'info', title, text = '', hint = '', timeout, action = null }) {
  const container = ensureHost();
  let timer = 0;
  let closed = false;

  const close = () => {
    if (closed) return;
    closed = true;
    clearTimeout(timer);
    el.classList.add('is-leaving');
    setTimeout(() => el.remove(), 180);
  };

  const el = h(
    'div',
    { class: `toast ${type}`, role: type === 'err' ? 'alert' : 'status', testid: `toast-${type}` },
    icon(ICON[type] || 'info'),
    h(
      'div',
      { class: 'toast-body' },
      title ? h('div', { class: 'toast-title' }, title) : null,
      text ? h('div', { class: 'toast-text' }, text) : null,
      hint ? h('div', { class: 'toast-hint' }, hint) : null,
      action
        ? h(
            'div',
            { class: 'note-actions' },
            h(
              'button',
              {
                class: 'btn small',
                type: 'button',
                onClick: () => {
                  close();
                  action.onClick();
                },
              },
              action.label,
            ),
          )
        : null,
    ),
    h('button', { class: 'btn ghost small icon-only', type: 'button', 'aria-label': 'Dismiss', onClick: close }, icon('x', 's')),
  );

  const arm = () => {
    const ms = timeout === undefined ? TIMEOUT[type] || 6000 : timeout;
    if (ms > 0) timer = setTimeout(close, ms);
  };
  el.addEventListener('mouseenter', () => clearTimeout(timer));
  el.addEventListener('mouseleave', arm);
  arm();

  container.appendChild(el);
  while (container.children.length > 5) container.firstElementChild.remove();
  return close;
}

/** Show an API error with its message and hint. Cancelled requests are ignored. */
export function toastError(error, title = 'The request failed') {
  const err = toApiError(error);
  if (err.isAborted) return () => {};
  return toast({ type: 'err', title, text: err.message, hint: err.hint || '' });
}
