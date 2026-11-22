// One floating tooltip for every element that carries a data-tip attribute.

import { h } from '../dom.js';

let tip = null;
let current = null;
let showTimer = 0;

function ensure() {
  if (!tip) {
    tip = h('div', { class: 'tooltip', role: 'tooltip' });
    document.body.appendChild(tip);
  }
  return tip;
}

function place(target) {
  const el = ensure();
  // inside a modal dialog the tooltip must live in the dialog (top layer), otherwise it is hidden
  const host = target.closest('dialog[open]') || document.body;
  if (el.parentNode !== host) host.appendChild(el);
  el.textContent = target.getAttribute('data-tip') || '';
  el.classList.remove('is-visible');
  el.style.left = '0px';
  el.style.top = '0px';
  const rect = target.getBoundingClientRect();
  const box = el.getBoundingClientRect();
  const gap = 8;
  let top = rect.top - box.height - gap;
  if (top < 4) top = rect.bottom + gap;
  let left = rect.left + rect.width / 2 - box.width / 2;
  left = Math.max(6, Math.min(left, window.innerWidth - box.width - 6));
  el.style.left = `${Math.round(left)}px`;
  el.style.top = `${Math.round(top)}px`;
  el.classList.add('is-visible');
}

function hide() {
  clearTimeout(showTimer);
  current = null;
  if (tip) tip.classList.remove('is-visible');
}

function show(target, delay) {
  if (current === target) return;
  hide();
  current = target;
  showTimer = setTimeout(() => {
    if (current === target && target.isConnected) place(target);
  }, delay);
}

export function installTooltips() {
  document.addEventListener('mouseover', (event) => {
    const target = event.target instanceof Element ? event.target.closest('[data-tip]') : null;
    if (target) show(target, 350);
    else if (current) hide();
  });
  document.addEventListener('focusin', (event) => {
    const target = event.target instanceof Element ? event.target.closest('[data-tip]') : null;
    if (target && target.matches(':focus-visible')) show(target, 100);
  });
  document.addEventListener('focusout', hide);
  document.addEventListener('pointerdown', hide, true);
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') hide();
  });
  window.addEventListener('scroll', hide, true);
}
