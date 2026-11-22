// The application frame: top bar with the system strip, workflow rail, context header naming
// the selected video(s), global banners and the area the views are mounted in.

import { h, s, fill, createScope, setPercent } from './dom.js';
import { icon } from './icons.js';
import { store, config, selectedVideos, activeVideo, subjectFor, runningJob, anyModelReady, currentModel } from './state.js';
import { describeSubject } from './config.js';
import { formatFps, formatDuration, formatGbPair, formatMb, formatPercent, isDone, isNumber, plural } from './format.js';
import { loadModels, loadSystem, navigate, setTheme } from './actions.js';
import { refreshAll } from './live.js';
import { button, kv } from './components/controls.js';
import { openModal } from './components/modal.js';
import { sourceView } from './views/source.js';
import { subjectView } from './views/subject.js';
import { settingsView } from './views/settings.js';
import { processView } from './views/process.js';
import { resultsView } from './views/results.js';
import { modelsView } from './views/models.js';

const VIEW_FACTORIES = {
  source: sourceView,
  subject: subjectView,
  settings: settingsView,
  process: processView,
  results: resultsView,
  models: modelsView,
};

const STEPS = [
  { view: 'source', label: 'Source', step: 1 },
  { view: 'subject', label: 'Subject', step: 2 },
  { view: 'settings', label: 'Settings', step: 3 },
  { view: 'process', label: 'Process', step: 4 },
  { view: 'results', label: 'Results', step: 5 },
];

function brandMark() {
  return s(
    'svg',
    { class: 'brand-mark', viewBox: '0 0 32 32', 'aria-hidden': 'true' },
    s('rect', { x: 3, y: 5, width: 26, height: 22, rx: 4, fill: '#00d26a' }),
    s('circle', { cx: 16, cy: 13, r: 4.2, fill: '#0d0f12' }),
    s('path', { d: 'M8.500 27c.700-4.800 3.700-7.300 7.500-7.300s6.800 2.500 7.500 7.300z', fill: '#0d0f12' }),
  );
}

// ------------------------------------------------------------------ system strip

function meter() {
  const bar = h('span');
  const el = h('span', { class: 'syschip-meter', 'aria-hidden': 'true' }, bar);
  return {
    el,
    set(percent) {
      setPercent(el, isNumber(percent) ? percent : 0);
      el.classList.toggle('warn', isNumber(percent) && percent >= 85 && percent < 95);
      el.classList.toggle('err', isNumber(percent) && percent >= 95);
    },
  };
}

function primaryGpu(system) {
  return system ? (system.devices || []).find((device) => device.type !== 'cpu') || null : null;
}

function systemDetails(state) {
  const { system, stats } = state;
  if (!system) {
    return [h('p', { class: 'muted' }, state.systemError ? state.systemError.message : 'System information has not been loaded yet.')];
  }
  const ffmpeg = system.ffmpeg || {};
  const devices = (system.devices || []).map((device) =>
    h(
      'tr',
      null,
      h('td', { class: 'mono' }, device.id),
      h('td', null, device.name),
      h('td', { class: 'num' }, isNumber(device.total_memory_mb) ? `${formatMb(device.free_memory_mb)} free of ${formatMb(device.total_memory_mb)}` : '–'),
      h('td', null, device.available ? device.detail : `Not usable: ${device.detail}`),
    ),
  );
  return [
    kv([
      ['Application', `Video Matting Tool ${system.version || ''}`],
      ['Platform', system.platform],
      ['Processor', system.cpu ? `${system.cpu} (${system.cpu_threads} threads)` : null],
      ['Memory', isNumber(system.ram_total_mb) ? `${formatMb(system.ram_total_mb)} installed` : null],
      ['Python', system.python],
      ['PyTorch', system.torch],
      ['CUDA runtime', system.cuda_runtime || 'not available'],
      ['cuDNN', system.cudnn ? String(system.cudnn) : null],
      ['FFmpeg', ffmpeg.available ? `${ffmpeg.version}` : `not found${ffmpeg.error ? `: ${ffmpeg.error}` : ''}`],
      ['FFmpeg path', ffmpeg.available ? ffmpeg.path : null],
      ['Default device', system.default_device],
      ['GPU temperature', stats && isNumber(stats.gpu_temperature_c) ? `${Math.round(stats.gpu_temperature_c)} °C` : null],
      ['Data folder', system.data_dir],
      ['Models folder', system.models_dir],
    ]),
    h(
      'div',
      { class: 'table-wrap' },
      h(
        'table',
        { class: 'table' },
        h('thead', null, h('tr', null, h('th', null, 'Device'), h('th', null, 'Name'), h('th', { class: 'num' }, 'Memory'), h('th', null, 'Details'))),
        h('tbody', null, devices),
      ),
    ),
  ];
}

function openSystemDialog() {
  const body = h('div', { class: 'stack' });
  const stop = store.watch(
    (state) => [state.system, state.stats, state.systemError],
    () => fill(body, systemDetails(store.get())),
  );
  openModal({
    title: 'System',
    size: 'wide',
    testid: 'system-dialog',
    body,
    footer: [button({ label: 'Refresh', icon: 'refresh', onClick: () => refreshAll() })],
    onClose: stop,
  });
}

function systemStrip(scope) {
  const gpuName = h('span', { class: 'syschip-value syschip-name' });
  const gpuUtil = h('span', { class: 'syschip-value' });
  const gpuChip = h('button', { class: 'syschip', type: 'button', testid: 'chip-gpu', onClick: openSystemDialog }, icon('monitor'), h('span', { class: 'syschip-label' }, 'GPU'), gpuName, gpuUtil);

  const vramMeter = meter();
  const vramValue = h('span', { class: 'syschip-value' });
  const vramChip = h('button', { class: 'syschip', type: 'button', testid: 'chip-vram', onClick: openSystemDialog }, h('span', { class: 'syschip-label' }, 'VRAM'), vramValue, vramMeter.el);

  const ramMeter = meter();
  const ramValue = h('span', { class: 'syschip-value' });
  const ramChip = h('button', { class: 'syschip', type: 'button', testid: 'chip-ram', onClick: openSystemDialog }, icon('memory'), h('span', { class: 'syschip-label' }, 'RAM'), ramValue, ramMeter.el);

  const el = h('div', { class: 'sysstrip', testid: 'system-strip' }, gpuChip, vramChip, ramChip);

  scope.add(
    store.watch(
      (state) => [state.system, state.stats, state.systemError],
      ([system, stats, systemError]) => {
        const gpu = primaryGpu(system);
        const hasGpuStats = stats && isNumber(stats.gpu_memory_total_mb);
        // GPU chip
        gpuChip.classList.toggle('is-warn', !!systemError && !system);
        if (gpu) {
          gpuName.textContent = gpu.name;
          gpuChip.setAttribute('data-tip', `${gpu.name} (${gpu.id})${gpu.available ? '' : ' - not usable'}`);
        } else if (system) {
          gpuName.textContent = 'none (CPU only)';
          gpuChip.setAttribute('data-tip', 'No GPU was detected: processing runs on the CPU.');
        } else {
          gpuName.textContent = systemError ? 'unavailable' : '…';
          gpuChip.setAttribute('data-tip', systemError ? `System information is unavailable: ${systemError.message}` : 'Loading system information');
        }
        gpuUtil.textContent = stats && isNumber(stats.gpu_utilization) ? formatPercent(stats.gpu_utilization) : '';
        gpuUtil.hidden = !gpuUtil.textContent;
        // VRAM chip
        vramChip.hidden = !hasGpuStats;
        if (hasGpuStats) {
          vramValue.textContent = formatGbPair(stats.gpu_memory_used_mb, stats.gpu_memory_total_mb);
          vramMeter.set((stats.gpu_memory_used_mb / stats.gpu_memory_total_mb) * 100);
          vramChip.setAttribute('data-tip', 'Video memory in use (all applications) / installed');
        }
        // RAM chip
        const hasRam = stats && isNumber(stats.ram_total_mb) && isNumber(stats.ram_available_mb);
        ramChip.hidden = !hasRam;
        if (hasRam) {
          ramValue.textContent = formatGbPair(stats.ram_total_mb - stats.ram_available_mb, stats.ram_total_mb);
          ramMeter.set(isNumber(stats.ram_system_percent) ? stats.ram_system_percent : ((stats.ram_total_mb - stats.ram_available_mb) / stats.ram_total_mb) * 100);
          ramChip.setAttribute('data-tip', `System memory in use / installed${isNumber(stats.ram_used_mb) ? ` - this application: ${formatMb(stats.ram_used_mb)}` : ''}`);
        }
      },
    ),
  );
  return el;
}

// ------------------------------------------------------------------ rail

function railItem({ view, label, step = null, iconName = null }) {
  const sub = h('span', { class: 'rail-sub' });
  const badgeEl = h('span', { class: 'rail-badge', hidden: true });
  const el = h(
    'a',
    { class: 'rail-item', href: `#/${view}`, testid: `nav-${view}`, dataset: { view } },
    step ? h('span', { class: 'rail-step' }, String(step)) : icon(iconName),
    h('span', { class: 'rail-text' }, h('span', { class: 'rail-label' }, label), sub),
    badgeEl,
  );
  return {
    el,
    set({ subText = '', badge = null, badgeTone = '', done = false }) {
      sub.textContent = subText;
      el.setAttribute('data-tip', subText ? `${label}: ${subText}` : label);
      badgeEl.hidden = badge === null || badge === '';
      badgeEl.textContent = badge === null ? '' : String(badge);
      badgeEl.className = `rail-badge ${badgeTone}`;
      el.classList.toggle('is-done', done);
    },
  };
}

function versionsFooter(scope) {
  const row = (label) => {
    const value = h('span', null, '–');
    return { el: h('div', { class: 'rail-foot-row' }, h('span', null, label), value), value };
  };
  const app = row('Version');
  const torch = row('PyTorch');
  const cuda = row('CUDA');
  const ffmpeg = row('FFmpeg');
  const el = h('div', { class: 'rail-foot', testid: 'versions' }, app.el, torch.el, cuda.el, ffmpeg.el);
  scope.add(
    store.watch(
      (state) => [state.system, state.options],
      ([system, options]) => {
        const version = (system && system.version) || (options && options.version) || null;
        app.value.textContent = version || '–';
        torch.value.textContent = system && system.torch ? String(system.torch) : '–';
        cuda.value.textContent = system ? system.cuda_runtime || 'none' : '–';
        const ff = system ? system.ffmpeg : null;
        ffmpeg.value.classList.toggle('is-missing', !!ff && ff.available === false);
        if (!ff) ffmpeg.value.textContent = '–';
        else if (ff.available === false) ffmpeg.value.textContent = 'not found';
        else {
          const match = String(ff.version || '').match(/\d+(?:\.\d+)+/);
          ffmpeg.value.textContent = match ? match[0] : String(ff.version || '–');
          ffmpeg.value.setAttribute('data-tip', String(ff.version || ''));
        }
      },
    ),
  );
  return el;
}

function rail(scope) {
  const items = Object.fromEntries(STEPS.map((step) => [step.view, railItem(step)]));
  items.models = railItem({ view: 'models', label: 'Models', iconName: 'package' });

  const el = h(
    'nav',
    { class: 'rail', 'aria-label': 'Workflow', testid: 'main-nav' },
    h('div', { class: 'rail-heading' }, 'Workflow'),
    h(
      'div',
      { class: 'rail-list' },
      STEPS.map((step) => items[step.view].el),
    ),
    h('div', { class: 'rail-sep' }),
    h('div', { class: 'rail-list' }, items.models.el),
    versionsFooter(scope),
  );

  scope.add(
    store.watch(
      (state) => state.route.view,
      (view) => {
        for (const [name, item] of Object.entries(items)) {
          if (name === view) item.el.setAttribute('aria-current', 'page');
          else item.el.removeAttribute('aria-current');
        }
      },
    ),
  );

  scope.add(
    store.watch(
      (state) => [state.videos, state.selectedIds, state.activeId, state.subjects, state.configVersion, state.jobs, state.liveJob && state.liveJob.id, state.models, state.options],
      () => {
        const state = store.get();
        const videos = selectedVideos(state);
        const active = activeVideo(state);
        items.source.set({
          subText: videos.length === 0 ? 'No video selected' : videos.length === 1 ? videos[0].filename : `${videos.length} videos (batch)`,
          done: videos.length > 0,
        });
        items.subject.set({ subText: active ? describeSubject(subjectFor(state, active.id)) : 'Automatic' });
        const changed = config.changedPaths().length;
        const model = currentModel(state);
        items.settings.set({
          subText: !config.ready ? '' : `${model ? model.name : config.get('model')}${changed ? ` · ${plural(changed, 'change')}` : ''}`,
        });
        const jobs = state.jobs || [];
        const queued = jobs.filter((job) => job.status === 'queued').length;
        const running = runningJob(state);
        items.process.set({
          subText: running ? `Running: ${running.name}` : queued ? `${plural(queued, 'job')} queued` : 'Idle',
          badge: running || queued ? (running ? 1 : 0) + queued : null,
          badgeTone: 'is-active',
        });
        const finished = jobs.filter(isDone).length;
        items.results.set({ subText: finished ? `${plural(finished, 'result')}` : 'Nothing yet', badge: finished || null });
        const models = state.models;
        const ready = (models || []).filter((m) => m.status && m.status.ready).length;
        items.models.set({
          subText: models ? `${ready} of ${models.length} ready` : '',
          badge: models && ready === 0 ? '!' : null,
          badgeTone: 'is-warn',
        });
      },
    ),
  );
  return el;
}

// ------------------------------------------------------------------ context header

function describeVideo(video) {
  const meta = video.metadata || {};
  const v = meta.video || {};
  const parts = [];
  if (v.display_width && v.display_height) parts.push(`${v.display_width}×${v.display_height}`);
  if (isNumber(meta.frame_rate_float)) parts.push(formatFps(meta.frame_rate_float) + (meta.is_vfr ? ' VFR' : ''));
  const duration = isNumber(meta.video_duration) ? meta.video_duration : meta.duration;
  if (isNumber(duration)) parts.push(formatDuration(duration));
  if (v.codec) parts.push(v.codec.toUpperCase());
  return parts.join(' · ');
}

function contextHeader(scope) {
  const selection = h('div', { class: 'context-sel' });
  const jobSlot = h('div');
  const el = h('div', { class: 'context', testid: 'context-header' }, h('span', { class: 'context-label' }, 'Selected'), selection, jobSlot);

  scope.add(
    store.watch(
      (state) => [state.videos, state.selectedIds, state.activeId],
      () => {
        const state = store.get();
        const videos = selectedVideos(state);
        if (state.videos === null) {
          fill(selection, h('span', { class: 'context-none' }, 'Loading the library…'));
        } else if (videos.length === 0) {
          fill(
            selection,
            h('span', { class: 'context-none', testid: 'context-none' }, 'No video selected'),
            state.route.view === 'source' ? null : h('a', { href: '#/source', class: 'link-btn' }, 'Choose a video'),
          );
        } else if (videos.length === 1) {
          fill(
            selection,
            icon('film'),
            h('span', { class: 'context-name', testid: 'context-name' }, videos[0].filename),
            h('span', { class: 'context-meta' }, describeVideo(videos[0])),
          );
        } else {
          fill(
            selection,
            icon('layers'),
            h('span', { class: 'context-name', testid: 'context-name' }, `${videos.length} videos (batch)`),
            h('span', { class: 'context-meta' }, videos.map((video) => video.filename).join(', ')),
          );
        }
      },
    ),
  );

  scope.add(
    store.watch(
      (state) => {
        const job = runningJob(state);
        return [job ? job.id : null, job ? job.name : null, job && job.progress ? Math.floor(job.progress.overall_percent || 0) : null];
      },
      ([id, name, percent]) => {
        if (!id) {
          fill(jobSlot);
          return;
        }
        fill(
          jobSlot,
          h(
            'a',
            { class: 'context-job', href: `#/process/${encodeURIComponent(id)}`, testid: 'context-job', tip: 'Open the running job' },
            h('span', { class: 'pulse-dot' }),
            h('span', { class: 'truncate' }, name),
            h('span', { class: 'pct' }, `${percent}%`),
          ),
        );
      },
    ),
  );
  return el;
}

// ------------------------------------------------------------------ banners

function banner({ tone, iconName, title, text, actions = null, testid }) {
  return h(
    'div',
    { class: `banner ${tone}`, role: tone === 'err' ? 'alert' : 'status', testid },
    icon(iconName),
    h('div', { class: 'banner-text break' }, h('strong', null, title), text ? ` ${text}` : null),
    actions,
  );
}

function banners(scope) {
  const el = h('div', { class: 'banners', testid: 'banners' });
  scope.add(
    store.watch(
      (state) => [state.online, state.system, state.systemError, state.models, state.modelsError, state.route.view],
      () => {
        const state = store.get();
        const list = [];
        if (!state.online) {
          list.push(
            banner({
              tone: 'err',
              iconName: 'alert',
              title: 'The connection to the application was lost.',
              text: 'Retrying every few seconds. Check that the Video Matting Tool is still running.',
              testid: 'banner-offline',
            }),
          );
        }
        const ffmpeg = state.system ? state.system.ffmpeg : null;
        if (ffmpeg && ffmpeg.available === false) {
          list.push(
            banner({
              tone: 'err',
              iconName: 'alert',
              title: 'FFmpeg was not found.',
              text: [ffmpeg.error, ffmpeg.hint].filter(Boolean).join(' '),
              actions: button({ label: 'Check again', icon: 'refresh', size: 'small', onClick: () => refreshAll() }),
              testid: 'banner-ffmpeg',
            }),
          );
        }
        if (state.online && state.systemError && !state.system) {
          list.push(
            banner({
              tone: 'warn',
              iconName: 'alert',
              title: 'System information is unavailable.',
              text: [state.systemError.message, state.systemError.hint].filter(Boolean).join(' '),
              actions: button({ label: 'Retry', icon: 'refresh', size: 'small', onClick: () => loadSystem() }),
              testid: 'banner-system',
            }),
          );
        }
        if (state.online && state.modelsError && !state.models) {
          list.push(
            banner({
              tone: 'warn',
              iconName: 'alert',
              title: 'The model list could not be loaded.',
              text: [state.modelsError.message, state.modelsError.hint].filter(Boolean).join(' '),
              actions: button({ label: 'Retry', icon: 'refresh', size: 'small', onClick: () => loadModels() }),
              testid: 'banner-models-error',
            }),
          );
        }
        if (state.models && !anyModelReady(state)) {
          list.push(
            banner({
              tone: 'warn',
              iconName: 'package',
              title: 'No model is ready to use.',
              text: state.models.length
                ? 'Download the weights of a model (or install its package) before starting a job.'
                : 'No model is registered in this installation.',
              actions: state.route.view === 'models' ? null : button({ label: 'Open Models', iconRight: 'arrow-right', size: 'small', onClick: () => navigate('models'), testid: 'banner-open-models' }),
              testid: 'banner-no-model',
            }),
          );
        }
        fill(el, list);
      },
    ),
  );
  return el;
}

// ------------------------------------------------------------------ shell

export function createShell() {
  const scope = createScope();
  const viewHost = h('main', { class: 'view', id: 'view', tabIndex: -1, testid: 'view' });
  let viewScope = null;

  const themeButton = h('button', { class: 'btn ghost icon-only', type: 'button', testid: 'theme-toggle', onClick: () => setTheme(store.get().theme === 'dark' ? 'light' : 'dark') });
  scope.add(
    store.watch(
      (state) => state.theme,
      (theme) => {
        fill(themeButton, icon(theme === 'dark' ? 'sun' : 'moon'));
        const label = theme === 'dark' ? 'Switch to the light theme' : 'Switch to the dark theme';
        themeButton.setAttribute('aria-label', label);
        themeButton.setAttribute('data-tip', label);
      },
    ),
  );

  const versionPill = h('span', { class: 'brand-version', hidden: true, testid: 'app-version' });
  scope.add(
    store.watch(
      (state) => (state.system && state.system.version) || (state.options && state.options.version) || null,
      (version) => {
        versionPill.hidden = !version;
        versionPill.textContent = version ? `v${version}` : '';
      },
    ),
  );

  const el = h(
    'div',
    { class: 'app', testid: 'app' },
    h(
      'header',
      { class: 'topbar' },
      h('a', { class: 'brand', href: '#/source', testid: 'brand' }, brandMark(), h('span', { class: 'brand-name' }, 'Video Matting Tool'), versionPill),
      h('div', { class: 'topbar-spacer' }),
      systemStrip(scope),
      themeButton,
    ),
    rail(scope),
    h('div', { class: 'main' }, contextHeader(scope), banners(scope), viewHost),
  );

  scope.add(
    store.watch(
      (state) => state.route.view,
      (view) => {
        if (viewScope) viewScope.dispose();
        viewScope = createScope();
        viewHost.dataset.view = view;
        let content;
        try {
          content = VIEW_FACTORIES[view](viewScope);
        } catch (error) {
          console.error(error);
          content = h('div', { class: 'note err' }, icon('x-circle'), h('div', { class: 'note-body' }, h('div', { class: 'note-title' }, 'This view could not be displayed.'), h('div', null, String(error && error.message ? error.message : error))));
        }
        fill(viewHost, content);
        viewHost.scrollTop = 0;
        const titles = { source: 'Source', subject: 'Subject', settings: 'Settings', process: 'Process', results: 'Results', models: 'Models' };
        document.title = `${titles[view]} · Video Matting Tool`;
      },
    ),
  );

  return el;
}
