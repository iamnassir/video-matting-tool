// A minimal observable store. State is replaced immutably; listeners are notified once per
// micro-task so several updates in a row cause a single re-render.

export function shallowEqual(a, b) {
  if (Object.is(a, b)) return true;
  if (typeof a !== 'object' || typeof b !== 'object' || a === null || b === null) return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const keysA = Object.keys(a);
  const keysB = Object.keys(b);
  if (keysA.length !== keysB.length) return false;
  for (const key of keysA) {
    if (!Object.prototype.hasOwnProperty.call(b, key) || !Object.is(a[key], b[key])) return false;
  }
  return true;
}

export function createStore(initialState) {
  let state = initialState;
  const listeners = new Set();
  let scheduled = false;

  function flush() {
    scheduled = false;
    for (const listener of [...listeners]) {
      try {
        listener(state);
      } catch (error) {
        console.error(error);
      }
    }
  }

  return {
    get() {
      return state;
    },

    /** Merge a partial state (or the result of an updater function) into the state. */
    set(patch) {
      const next = typeof patch === 'function' ? patch(state) : patch;
      if (!next) return;
      let changed = false;
      for (const key of Object.keys(next)) {
        if (!Object.is(state[key], next[key])) {
          changed = true;
          break;
        }
      }
      if (!changed) return;
      state = { ...state, ...next };
      if (!scheduled) {
        scheduled = true;
        queueMicrotask(flush);
      }
    },

    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },

    /**
     * Call `effect(value, state)` now and whenever `selector(state)` changes. Selectors may return
     * a fresh array / object of values: they are compared shallowly.
     */
    watch(selector, effect, options = {}) {
      let current = selector(state);
      if (options.immediate !== false) effect(current, state);
      const listener = (s) => {
        const next = selector(s);
        if (shallowEqual(next, current)) return;
        current = next;
        effect(next, s);
      };
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}
