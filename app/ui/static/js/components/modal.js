// Modal dialogs built on the native <dialog> element (focus trapping and Escape come for free).

import { h } from '../dom.js';
import { icon } from '../icons.js';

/**
 * Open a modal. `body` and `footer` are nodes (or arrays of nodes). Returns {dialog, close, closed}
 * where `closed` resolves with the value passed to close() (undefined when dismissed).
 */
export function openModal({ title, body, footer = null, size = '', testid = 'modal', onClose = null, dismissible = true }) {
  let resolveClosed;
  const closed = new Promise((resolve) => {
    resolveClosed = resolve;
  });
  let result;
  let done = false;

  const close = (value) => {
    if (done) return;
    result = value;
    dialog.close();
  };

  const dialog = h(
    'dialog',
    { class: ['modal', size], testid, 'aria-label': title },
    h(
      'div',
      { class: 'modal-head' },
      h('h2', null, title),
      dismissible
        ? h('button', { class: 'btn ghost icon-only', type: 'button', 'aria-label': 'Close', testid: 'modal-close', onClick: () => close() }, icon('x'))
        : null,
    ),
    h('div', { class: 'modal-body' }, body),
    footer ? h('div', { class: 'modal-foot' }, footer) : null,
  );

  dialog.addEventListener('cancel', (event) => {
    if (!dismissible) event.preventDefault();
  });
  // a click on the backdrop (the dialog element itself, outside its content box) dismisses
  let downOnBackdrop = false;
  dialog.addEventListener('pointerdown', (event) => {
    downOnBackdrop = event.target === dialog;
  });
  dialog.addEventListener('click', (event) => {
    if (dismissible && downOnBackdrop && event.target === dialog) close();
  });
  dialog.addEventListener('close', () => {
    done = true;
    dialog.remove();
    if (onClose) onClose(result);
    resolveClosed(result);
  });

  document.body.appendChild(dialog);
  dialog.showModal();
  return { dialog, close, closed };
}

/** Ask a yes / no question. Resolves to true when confirmed. */
export function confirmDialog({ title, message, detail = '', confirmLabel = 'Confirm', cancelLabel = 'Cancel', danger = false }) {
  let modal = null;
  const confirm = h(
    'button',
    { class: ['btn', danger ? 'danger solid' : 'primary'], type: 'button', testid: 'confirm-ok', onClick: () => modal.close(true) },
    confirmLabel,
  );
  const cancel = h('button', { class: 'btn', type: 'button', testid: 'confirm-cancel', onClick: () => modal.close(false) }, cancelLabel);
  modal = openModal({
    title,
    testid: 'confirm-dialog',
    body: [h('p', { class: 'break' }, message), detail ? h('p', { class: 'muted break' }, detail) : null],
    footer: [cancel, confirm],
  });
  confirm.focus();
  return modal.closed.then((value) => value === true);
}
