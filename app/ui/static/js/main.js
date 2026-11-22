// Entry point: restores remembered choices, wires the router, builds the shell and starts the
// live updates.

import { store, config, VIEWS } from './state.js';
import { persist } from './persist.js';
import { installTooltips } from './components/tooltip.js';
import { createShell } from './shell.js';
import { refreshAll, startLiveUpdates } from './live.js';
import { resumeDownloadWatch } from './actions.js';

function parseHash() {
  const match = window.location.hash.match(/^#\/([a-z]+)(?:\/(.+))?$/);
  if (match && VIEWS.includes(match[1])) {
    let param = null;
    try {
      param = match[2] ? decodeURIComponent(match[2]) : null;
    } catch {
      param = null;
    }
    return { view: match[1], param };
  }
  return { view: 'source', param: null };
}

function applyRoute() {
  const route = parseHash();
  const patch = { route };
  if (route.view === 'results' && route.param) patch.resultJobId = route.param;
  if (route.view === 'process' && route.param) patch.focusJobId = route.param;
  store.set(patch);
}

function restore() {
  const theme = persist.get('theme', 'dark') === 'light' ? 'light' : 'dark';
  document.documentElement.dataset.theme = theme;
  const selectedIds = persist.get('selectedIds', []);
  const subjects = persist.get('subjects', {});
  store.set({
    theme,
    selectedIds: Array.isArray(selectedIds) ? selectedIds.filter((id) => typeof id === 'string') : [],
    activeId: typeof persist.get('activeId') === 'string' ? persist.get('activeId') : null,
    subjects: subjects && typeof subjects === 'object' ? subjects : {},
  });
  const overrides = persist.get('overrides', {});
  if (overrides && typeof overrides === 'object') config.overrides = { ...overrides };
}

function remember() {
  store.watch(
    (state) => [state.selectedIds, state.activeId, state.subjects, state.configVersion],
    () => {
      const state = store.get();
      persist.assign({
        selectedIds: state.selectedIds,
        activeId: state.activeId,
        subjects: state.subjects,
        overrides: config.overrides,
      });
    },
    { immediate: false },
  );
}

function boot() {
  restore();
  applyRoute();
  window.addEventListener('hashchange', applyRoute);

  const shell = createShell();
  document.getElementById('app').replaceChildren(shell);
  installTooltips();
  remember();

  refreshAll();
  startLiveUpdates();
  resumeDownloadWatch();
}

boot();
