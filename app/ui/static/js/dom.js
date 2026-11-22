// Small DOM toolkit. Strings always become text nodes or attribute values: nothing in this file
// (or anywhere else in the interface) parses HTML, so user supplied names can never inject markup.

const SVG_NS = 'http://www.w3.org/2000/svg';

// Attributes that must be set with setAttribute (read-only or non-reflecting DOM properties).
const ATTRIBUTE_ONLY = new Set(['list', 'form', 'role', 'for', 'viewBox', 'd', 'fill', 'stroke', 'points']);

function appendChildren(parent, children) {
  for (const child of children) {
    if (child === null || child === undefined || child === false || child === true) continue;
    if (Array.isArray(child)) appendChildren(parent, child);
    else if (child instanceof Node) parent.appendChild(child);
    else parent.appendChild(document.createTextNode(String(child)));
  }
}

function applyProps(el, props, isSvg) {
  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class' || key === 'className') {
      const text = Array.isArray(value) ? value.filter(Boolean).join(' ') : String(value);
      if (text) el.setAttribute('class', text);
    } else if (key === 'dataset') {
      for (const [name, v] of Object.entries(value)) {
        if (v !== null && v !== undefined && v !== false) el.dataset[name] = String(v);
      }
    } else if (key === 'style') {
      // only used for CSS custom properties that carry dynamic values (progress, colours)
      for (const [name, v] of Object.entries(value)) {
        if (v !== null && v !== undefined) el.style.setProperty(name, String(v));
      }
    } else if (key === 'testid') {
      el.setAttribute('data-testid', String(value));
    } else if (key === 'tip') {
      el.setAttribute('data-tip', String(value));
    } else if (key.startsWith('on') && typeof value === 'function') {
      el.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (isSvg || ATTRIBUTE_ONLY.has(key) || key.startsWith('aria-') || key.startsWith('data-')) {
      el.setAttribute(key, value === true ? '' : String(value));
    } else if (key in el) {
      el[key] = value;
    } else {
      el.setAttribute(key, value === true ? '' : String(value));
    }
  }
}

/** Create an HTML element: h('div', {class: 'x', onClick: fn}, 'text', childNode, [more]). */
export function h(tag, props, ...children) {
  const el = document.createElement(tag);
  if (props) applyProps(el, props, false);
  appendChildren(el, children);
  return el;
}

/** Create an SVG element. */
export function s(tag, props, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  if (props) applyProps(el, props, true);
  appendChildren(el, children);
  return el;
}

/** Replace the children of *el*. */
export function fill(el, ...children) {
  const fragment = document.createDocumentFragment();
  appendChildren(fragment, children);
  el.replaceChildren(fragment);
  return el;
}

/**
 * Keyed list rendering: `update(items, key, signature, render)` re-creates only the children
 * whose signature changed and keeps the others (with their images, focus and hover state).
 */
export function keyedList(container) {
  let cache = new Map();
  return function update(items, key, signature, render) {
    const next = new Map();
    const nodes = items.map((item) => {
      const k = key(item);
      const sig = signature(item);
      let entry = cache.get(k);
      if (!entry || entry.sig !== sig) entry = { sig, el: render(item) };
      next.set(k, entry);
      return entry.el;
    });
    cache = next;
    let cursor = container.firstChild;
    for (const node of nodes) {
      if (node === cursor) cursor = cursor.nextSibling;
      else container.insertBefore(node, cursor);
    }
    while (cursor) {
      const dead = cursor;
      cursor = cursor.nextSibling;
      dead.remove();
    }
  };
}

/** addEventListener that returns the function removing the listener again. */
export function on(target, type, handler, options) {
  target.addEventListener(type, handler, options);
  return () => target.removeEventListener(type, handler, options);
}

/** Collects clean-up functions of a view / component so they can be released together. */
export function createScope() {
  const disposers = [];
  let disposed = false;
  return {
    get disposed() {
      return disposed;
    },
    add(disposer) {
      if (typeof disposer !== 'function') return disposer;
      if (disposed) disposer();
      else disposers.push(disposer);
      return disposer;
    },
    on(target, type, handler, options) {
      return this.add(on(target, type, handler, options));
    },
    interval(fn, ms) {
      const id = setInterval(fn, ms);
      return this.add(() => clearInterval(id));
    },
    timeout(fn, ms) {
      const id = setTimeout(fn, ms);
      return this.add(() => clearTimeout(id));
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      while (disposers.length) {
        const fn = disposers.pop();
        try {
          fn();
        } catch (error) {
          console.error(error);
        }
      }
    },
  };
}

export function debounce(fn, ms) {
  let timer = 0;
  const wrapped = (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
  wrapped.cancel = () => clearTimeout(timer);
  wrapped.flush = (...args) => {
    clearTimeout(timer);
    fn(...args);
  };
  return wrapped;
}

export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

let uidCounter = 0;
export function uid(prefix = 'id') {
  uidCounter += 1;
  return `${prefix}-${uidCounter}`;
}

/** Set a CSS custom property that carries a percentage (0..100) for bars and meters. */
export function setPercent(el, percent) {
  const value = Number.isFinite(percent) ? clamp(percent, 0, 100) : 0;
  el.style.setProperty('--p', value.toFixed(2));
}

/** True when keyboard input currently belongs to a text-like control. */
export function isEditableTarget(target) {
  if (!(target instanceof Element)) return false;
  if (target.closest('[contenteditable=""], [contenteditable="true"]')) return true;
  const tag = target.tagName;
  if (tag === 'TEXTAREA' || tag === 'SELECT') return true;
  if (tag === 'INPUT') {
    const type = (target.getAttribute('type') || 'text').toLowerCase();
    return !['checkbox', 'radio', 'range', 'button', 'submit', 'color', 'file'].includes(type);
  }
  return false;
}
