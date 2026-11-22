// Remembers the user's choices (selection, changed settings, subject selections, view preferences)
// in localStorage so a reload continues where the user left off. Server data is never cached here.

const KEY = 'vmt.ui.v1';

function read() {
  try {
    const raw = window.localStorage.getItem(KEY);
    const data = raw ? JSON.parse(raw) : {};
    return data && typeof data === 'object' ? data : {};
  } catch {
    return {};
  }
}

let cache = read();
let timer = 0;

function write() {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(cache));
  } catch {
    // storage full or disabled: the interface simply does not remember
  }
}

function schedule() {
  clearTimeout(timer);
  timer = setTimeout(write, 250);
}

export const persist = {
  get(key, fallback = null) {
    return Object.prototype.hasOwnProperty.call(cache, key) ? cache[key] : fallback;
  },
  set(key, value) {
    cache[key] = value;
    schedule();
  },
  /** Merge several keys. */
  assign(values) {
    Object.assign(cache, values);
    schedule();
  },
  flush() {
    clearTimeout(timer);
    write();
  },
};

window.addEventListener('pagehide', () => persist.flush());
