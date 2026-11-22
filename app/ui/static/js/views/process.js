// Process: start jobs for the selected videos, follow the job that is being processed (measured
// progress and a live picture of what is being produced) and manage the queue and the history.

import { h, fill, createScope, keyedList, setPercent } from '../dom.js';
import { icon } from '../icons.js';
import { api } from '../api.js';
import { store, config, currentModel, jobById, runningJob, selectedVideos, subjectFor } from '../state.js';
import { codecInfo, describeSubject, settingLabel, subjectProblem } from '../config.js';
import {
  formatDateTime,
  formatDuration,
  formatFps,
  formatGbPair,
  formatInt,
  formatMb,
  formatPercent,
  formatRate,
  formatSpan,
  isActive,
  isDone,
  isNumber,
  isResumable,
  plural,
  stageLabel,
  statusInfo,
} from '../format.js';
import { buildStartRequests, focusJob, loadJobs, navigate, setActive, showResults, startProcessing } from '../actions.js';
import { persist } from '../persist.js';
import { bullets, busyWhile, button, card, disclosure, emptyState, errorNote, note, progressBar, segmented, skeletonCards } from '../components/controls.js';
import { modelState } from '../components/modelinfo.js';
import {
  cancelWithConfirm,
  deleteWithConfirm,
  jobDurationText,
  jobErrorNote,
  jobSettingsLine,
  jobStatusBadge,
  openLogDialog,
  resumeWithFeedback,
} from '../components/jobparts.js';

// ------------------------------------------------------------------ what "Start" would do

function videoSummary(video) {
  const meta = video.metadata || {};
  const stream = meta.video || {};
  const duration = isNumber(meta.video_duration) ? meta.video_duration : meta.duration;
  return [
    stream.display_width && stream.display_height ? `${stream.display_width}×${stream.display_height}` : null,
    isNumber(meta.frame_rate_float) ? formatFps(meta.frame_rate_float) : null,
    isNumber(duration) ? formatDuration(duration) : null,
    isNumber(meta.frame_count) ? `${formatInt(meta.frame_count)} frames` : null,
  ]
    .filter(Boolean)
    .join(' · ');
}

/** Reasons a start would certainly fail (blocking) and things the user should know (warnings). */
function startChecks(state) {
  const blocking = [];
  const warnings = [];
  const videos = selectedVideos(state);
  const toModels = { label: 'Open Models', onClick: () => navigate('models') };
  const toSettings = { label: 'Open Settings', onClick: () => navigate('settings') };

  if (state.system && state.system.ffmpeg && state.system.ffmpeg.available === false) {
    blocking.push({ text: 'FFmpeg was not found: videos can neither be read nor written.', hint: state.system.ffmpeg.hint || '' });
  }
  for (const video of videos) {
    const problem = subjectProblem(subjectFor(state, video.id));
    if (problem) {
      blocking.push({
        text: `${video.filename}: the subject selection is incomplete. ${problem}`,
        action: {
          label: 'Open Subject',
          onClick: () => {
            setActive(video.id);
            navigate('subject');
          },
        },
      });
    }
  }
  if (state.models && config.ready) {
    const model = currentModel(state);
    if (!model || model.kind === 'auxiliary') {
      blocking.push({ text: `The selected model “${config.get('model')}” does not exist in this installation.`, action: toSettings });
    } else {
      const status = modelState(model);
      if (status.id === 'package') {
        blocking.push({ text: `${model.name} cannot run: a Python package it needs is not installed.`, hint: 'Models shows the command that installs it.', action: toModels });
      } else if (status.id === 'weights') {
        warnings.push({
          text: `The weight files of ${model.name} are not on this computer.`,
          hint: 'The job fails with a clear message unless “Download missing weights” allows it to download them (Settings, Advanced).',
          action: toModels,
        });
      }
      const license = model.license || {};
      if (config.get('commercial_use') === true && license.commercial_use === 'prohibited') {
        blocking.push({ text: `${model.name} is licensed under ${license.name}, which prohibits commercial use, and this job is marked as commercial.`, action: toSettings });
      }
    }
  }
  const codec = codecInfo(state.options, config.get('encoding.codec'));
  if (codec && codec.available === false) {
    warnings.push({ text: `${codec.label} is not available in this FFmpeg build: the job would fail when encoding starts.`, action: toSettings });
  }
  return { blocking, warnings };
}

function checkList(items, tone, testid) {
  return h(
    'div',
    { class: 'check-notes', testid },
    items.map((item) =>
      note({
        tone,
        text: item.text,
        hint: item.hint || '',
        actions: item.action ? [button({ label: item.action.label, size: 'small', iconRight: 'arrow-right', onClick: item.action.onClick })] : null,
      }),
    ),
  );
}

function startPanel(scope) {
  const body = h('div', { class: 'stack' });
  const start = button({ label: 'Start processing', icon: 'play', variant: 'primary', size: 'large', testid: 'start-job' });
  start.addEventListener('click', () => busyWhile(start, () => startProcessing()));
  const startLabel = start.lastChild; // the text node of the button

  function render() {
    const state = store.get();
    const videos = selectedVideos(state);
    if (state.videos === null || !config.ready) {
      fill(body, state.videosError || state.optionsError ? errorNote(state.videosError || state.optionsError, { title: 'The data for a new job could not be loaded' }) : skeletonCards(1));
      return;
    }
    if (!videos.length) {
      fill(
        body,
        emptyState({
          iconName: 'film',
          title: 'No video selected',
          text: 'Select one video for a single job, or several for a batch.',
          actions: [button({ label: 'Choose a video', icon: 'film', variant: 'primary', onClick: () => navigate('source'), testid: 'start-choose-video' })],
          compact: true,
          plain: true,
          testid: 'start-empty',
        }),
      );
      return;
    }
    const checks = startChecks(state);
    const requests = buildStartRequests(state);
    const model = currentModel(state);
    const changed = config.changedPaths().length;
    startLabel.textContent = videos.length === 1 ? 'Start processing' : `Start batch (${videos.length} videos)`;
    start.disabled = checks.blocking.length > 0 || state.starting;

    const items = videos.map((video) => {
      const subject = subjectFor(state, video.id);
      const problem = subjectProblem(subject);
      return h(
        'li',
        { class: 'start-item', testid: 'start-video', dataset: { videoId: video.id } },
        icon('film'),
        h('div', { class: 'start-item-main' }, h('span', { class: 'start-item-name truncate', tip: video.filename }, video.filename), h('span', { class: 'start-item-meta num' }, videoSummary(video))),
        h(
          'button',
          {
            class: ['start-item-subject', problem && 'is-problem'],
            type: 'button',
            tip: 'Edit the subject selection of this video',
            onClick: () => {
              setActive(video.id);
              navigate('subject');
            },
          },
          icon('target', 's'),
          h('span', { class: 'truncate' }, `Subject: ${describeSubject(subject)}`),
        ),
      );
    });

    const error = state.startErrors;
    let errorBlock = null;
    if (error && error.error) {
      const fieldLines = Object.entries(error.fields || {}).map(([path, text]) => `${settingLabel(path)}: ${text}`);
      const validation = fieldLines.length > 0 || (error.general || []).length > 0;
      errorBlock = validation
        ? note({
            tone: 'err',
            title: 'The server did not accept the settings',
            children: bullets([...(error.general || []), ...fieldLines]),
            hint: error.queued ? `${plural(error.queued, 'job')} had already been queued before the error.` : '',
            actions: [button({ label: 'Open Settings', size: 'small', iconRight: 'arrow-right', onClick: () => navigate('settings'), testid: 'start-error-settings' })],
            testid: 'start-error',
          })
        : errorNote(error.error, { title: 'The job could not be started', testid: 'start-error' });
    }

    fill(
      body,
      h('ul', { class: 'start-list' }, items),
      h(
        'div',
        { class: 'start-settings' },
        icon('sliders'),
        h('span', { class: 'break' }, [model ? model.name : config.get('model'), jobLikeSettings(state)].filter(Boolean).join(' · ')),
        changed ? h('span', { class: 'faint' }, `${plural(changed, 'setting')} changed`) : h('span', { class: 'faint' }, 'default settings'),
        h('a', { class: 'link-btn', href: '#/settings', testid: 'start-settings-link' }, 'Change'),
      ),
      requests.length > 1
        ? note({ tone: 'info', text: `Each video is queued as its own job (${requests.length} requests) because their subject selections differ.`, testid: 'start-separate' })
        : null,
      checks.blocking.length ? checkList(checks.blocking, 'err', 'start-blocking') : null,
      checks.warnings.length ? checkList(checks.warnings, 'warn', 'start-warnings') : null,
      errorBlock,
      h('div', { class: 'start-actions' }, start, h('span', { class: 'field-hint' }, 'Jobs are processed one at a time; further jobs wait in the queue.')),
    );
  }

  scope.add(
    store.watch(
      (state) => [state.videos, state.videosError, state.selectedIds, state.subjects, state.configVersion, state.models, state.options, state.optionsError, state.system, state.startErrors, state.starting],
      render,
    ),
  );
  return card({ title: 'New job', body, testid: 'start-panel' });
}

/** Quality, background and output of the current settings in a few words. */
function jobLikeSettings(state) {
  const quality = state.options ? state.options.quality_presets.find((item) => item.id === config.get('quality')) : null;
  const background = config.get('background') === 'custom' ? config.get('background_color') : config.get('background');
  const codec = codecInfo(state.options, config.get('encoding.codec'));
  return [quality ? quality.label : null, background ? `${background} background` : null, codec ? codec.label : 'automatic codec'].filter(Boolean).join(' · ');
}

// ------------------------------------------------------------------ live picture

const LIVE_TILE = { original: [0, 0], subject: [1, 0], matte: [0, 1], composite: [1, 1] };
const LIVE_VIEWS = [
  { value: 'original', label: 'Original' },
  { value: 'subject', label: 'Subject' },
  { value: 'matte', label: 'Matte' },
  { value: 'composite', label: 'Composite' },
  { value: 'grid', label: 'All four', icon: 'grid' },
];

/**
 * The picture the pipeline publishes while it renders: a 2 x 2 atlas (original, subject overlay,
 * matte, composite) fetched about once per second. Until the first frame exists the server
 * answers 204 and a placeholder says so.
 */
function livePicture(scope, getJob) {
  const canvas = h('canvas', { class: 'live-canvas', testid: 'live-canvas', hidden: true });
  const labels = h('div', { class: 'player-labels is-grid', 'aria-hidden': 'true', hidden: true }, ['original', 'subject', 'matte', 'composite'].map((name) => h('span', { class: `player-label is-${name}` }, name.charAt(0).toUpperCase() + name.slice(1))));
  const waiting = h('div', { class: 'live-waiting', testid: 'live-waiting', role: 'status' }, h('span', { class: 'spinner' }), h('span', null, 'The live picture appears as soon as the first frame has been processed.'));
  const stage = h('div', { class: 'live-stage' }, canvas, labels, waiting);
  const caption = h('span', { class: 'live-caption num faint', testid: 'live-frame' });
  let view = persist.get('liveView', 'composite');
  if (!LIVE_VIEWS.some((item) => item.value === view)) view = 'composite';
  let atlas = null;
  let version = null;
  let busy = false;
  let controller = null;

  function draw() {
    if (!atlas) return;
    const tileWidth = atlas.naturalWidth / 2;
    const tileHeight = atlas.naturalHeight / 2;
    const grid = view === 'grid';
    const width = grid ? atlas.naturalWidth : tileWidth;
    const height = grid ? atlas.naturalHeight : tileHeight;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const context = canvas.getContext('2d', { alpha: false });
    if (grid) context.drawImage(atlas, 0, 0);
    else {
      const [column, row] = LIVE_TILE[view];
      context.drawImage(atlas, column * tileWidth, row * tileHeight, tileWidth, tileHeight, 0, 0, tileWidth, tileHeight);
    }
    stage.style.setProperty('--ar', String(width / height));
    canvas.hidden = false;
    waiting.hidden = true;
    labels.hidden = !grid;
  }

  const views = segmented({
    options: LIVE_VIEWS,
    value: view,
    label: 'Live picture',
    testid: 'live-view',
    onChange: (next) => {
      view = next;
      persist.set('liveView', next);
      draw();
    },
  });

  async function poll() {
    const job = getJob();
    if (busy || !job || job.status !== 'running' || document.hidden) return;
    busy = true;
    controller = new AbortController();
    try {
      const result = await api.live(job, controller.signal);
      if (result && (result.version === null || result.version !== version)) {
        const url = URL.createObjectURL(result.blob);
        const image = new Image();
        image.onload = () => {
          URL.revokeObjectURL(url);
          if (scope.disposed) return;
          atlas = image;
          version = result.version;
          caption.textContent = isNumber(result.frameIndex) ? `Frame ${formatInt(result.frameIndex)}` : '';
          draw();
        };
        image.onerror = () => URL.revokeObjectURL(url);
        image.src = url;
      }
    } catch {
      // a missed picture is not worth a message: the next poll simply tries again
    } finally {
      busy = false;
    }
  }
  scope.interval(poll, 1000);
  // a poll in flight when the panel closes is left to finish (its picture is discarded): aborting
  // it would only show up as a failed request
  poll();

  return h('div', { class: 'live', testid: 'live' }, h('div', { class: 'live-head' }, h('span', { class: 'section-title' }, 'Live picture'), caption), stage, views.el);
}

// ------------------------------------------------------------------ monitor of one job

function tile(label, testid) {
  const number = h('span', { testid }, '–');
  const unit = h('small');
  const sub = h('div', { class: 'tile-sub', hidden: true });
  const el = h('div', { class: 'tile' }, h('div', { class: 'tile-label' }, label), h('div', { class: 'tile-value' }, number, unit), sub);
  return {
    el,
    /** `unitText` is only shown next to a real value, never next to the dash of a missing one. */
    set(text, subText = '', unitText = '') {
      number.textContent = text;
      unit.textContent = text === '–' ? '' : unitText;
      sub.textContent = subText;
      sub.hidden = !subText;
    },
  };
}

function meter(label, testid) {
  const bar = h('span');
  const track = h('div', { class: 'meter-bar' }, bar);
  const value = h('span', { class: 'meter-value num' });
  const el = h('div', { class: 'meter', testid, hidden: true }, h('span', { class: 'meter-label' }, label), track, value);
  return {
    el,
    /** percent null hides the meter: the value is not available on this system. */
    set(percent, text) {
      el.hidden = !isNumber(percent);
      if (!isNumber(percent)) return;
      setPercent(track, percent);
      track.classList.toggle('warn', percent >= 85 && percent < 95);
      track.classList.toggle('err', percent >= 95);
      value.textContent = text;
    },
  };
}

const FRAME_STAGES = ['render', 'backtrack'];

function runningPanel(scope, getJob) {
  const stage = h('span', { class: 'mon-stage', testid: 'job-stage' });
  const message = h('span', { class: 'mon-message break' });
  const percent = h('span', { class: 'mon-percent num', testid: 'job-percent' });
  const bar = progressBar({ large: true, testid: 'job-progress', label: 'Overall progress' });
  const tiles = {
    frame: tile('Frame', 'job-frame'),
    speed: tile('Processing', 'job-fps'),
    inference: tile('Inference', 'job-inference-fps'),
    eta: tile('Remaining', 'job-eta'),
    elapsed: tile('Elapsed', 'job-elapsed'),
    batch: tile('Batch size', 'job-batch'),
    device: tile('Device', 'job-device'),
    warnings: tile('Warnings', 'job-warnings'),
  };
  const meters = {
    gpu: meter('GPU load', 'meter-gpu'),
    vram: meter('VRAM', 'meter-vram'),
    ram: meter('System RAM', 'meter-ram'),
  };
  const stream = h('span', { class: 'mon-stream faint' });
  // elapsed time keeps running between progress events (a stage such as loading the model can
  // take a while without one): the last reported value plus the time since it arrived
  let clock = { reported: null, at: 0, stamp: null };

  function showElapsed() {
    const job = getJob();
    if (clock.reported === null || !job || job.status !== 'running') return;
    tiles.elapsed.set(formatSpan(clock.reported + (performance.now() - clock.at) / 1000));
  }
  scope.interval(showElapsed, 1000);

  function update(job, streamMode) {
    const p = job.progress || {};
    const framesStage = FRAME_STAGES.includes(p.stage);
    const stageText = stageLabel(p.stage || 'queued');
    stage.textContent = stageText;
    message.textContent = p.message && p.message !== stageText ? p.message : '';
    const overall = isNumber(p.overall_percent) ? p.overall_percent : null;
    bar.set(overall);
    percent.textContent = overall === null ? '' : formatPercent(overall, 1);

    // frame counters and rates describe frames only while frames are processed; other stages
    // (verification, for example) report their own progress as a percentage
    tiles.frame.set(framesStage && isNumber(p.total_frames) ? `${formatInt(p.frame)} / ${formatInt(p.total_frames)}` : '–', framesStage ? '' : isNumber(p.percent) && p.stage ? `this stage: ${formatPercent(p.percent)}` : '');
    tiles.speed.set(framesStage && isNumber(p.fps) ? formatRate(p.fps) : '–', 'end to end', 'fps');
    tiles.inference.set(isNumber(p.inference_fps) ? formatRate(p.inference_fps) : '–', 'model only', 'fps');
    tiles.eta.set(framesStage && isNumber(p.eta_seconds) ? formatSpan(p.eta_seconds) : '–');
    if (isNumber(p.elapsed_seconds)) {
      if (p.timestamp !== clock.stamp || clock.reported === null) clock = { reported: p.elapsed_seconds, at: performance.now(), stamp: p.timestamp };
      showElapsed();
    } else {
      tiles.elapsed.set('–');
    }
    tiles.batch.set(isNumber(p.batch_size) ? String(p.batch_size) : '–', '', p.batch_size === 1 ? 'frame per step' : 'frames per step');
    tiles.device.set(p.device || '–', p.gpu_name || '');
    tiles.warnings.set(isNumber(p.warnings) ? String(p.warnings) : '–');
    tiles.warnings.el.classList.toggle('is-warn', isNumber(p.warnings) && p.warnings > 0);

    meters.gpu.set(isNumber(p.gpu_utilization) ? p.gpu_utilization : null, formatPercent(p.gpu_utilization));
    const vram = isNumber(p.gpu_memory_used_mb) && isNumber(p.gpu_memory_total_mb) && p.gpu_memory_total_mb > 0;
    meters.vram.set(vram ? (p.gpu_memory_used_mb / p.gpu_memory_total_mb) * 100 : null, formatGbPair(p.gpu_memory_used_mb, p.gpu_memory_total_mb));
    meters.ram.set(isNumber(p.ram_system_percent) ? p.ram_system_percent : null, `${formatPercent(p.ram_system_percent)}${isNumber(p.ram_used_mb) ? ` · engine ${formatMb(p.ram_used_mb)}` : ''}`);

    stream.textContent = streamMode === 'events' ? 'Live updates over the event stream' : streamMode === 'polling' ? 'Live updates by polling every second' : '';
  }

  const el = h(
    'div',
    { class: 'mon-running', testid: 'job-running' },
    h('div', { class: 'mon-top' }, h('div', { class: 'mon-stage-row' }, h('span', { class: 'pulse-dot' }), stage, message, h('span', { class: 'grow' }), percent), bar.el),
    h(
      'div',
      { class: 'mon-grid' },
      livePicture(scope, getJob),
      h(
        'div',
        { class: 'stack' },
        h('div', { class: 'tiles mon-tiles' }, Object.values(tiles).map((item) => item.el)),
        h('div', { class: 'meters', testid: 'job-meters' }, Object.values(meters).map((item) => item.el)),
        stream,
      ),
    ),
  );
  return { el, update };
}

function finishedSummary(job) {
  const s = job.summary || {};
  const checksPassed = isNumber(s.checks_total) ? s.checks_total - (s.checks_failed || 0) - (s.checks_warned || 0) : null;
  const items = [
    ['Frames', isNumber(s.frames) ? formatInt(s.frames) : null, isNumber(s.frames_from_cache) && s.frames_from_cache > 0 ? `${formatInt(s.frames_from_cache)} from stored mattes` : '', ''],
    ['Total time', isNumber(s.seconds) ? formatSpan(s.seconds) : null, '', ''],
    ['Processing', isNumber(s.average_fps) ? formatRate(s.average_fps) : null, 'end to end', 'fps'],
    ['Inference', isNumber(s.average_inference_fps) ? formatRate(s.average_inference_fps) : null, 'model only', 'fps'],
    ['Device', s.device || null, '', ''],
    ['Quality checks', isNumber(s.checks_total) && s.checks_total > 0 ? `${checksPassed} / ${s.checks_total}` : null, [s.checks_warned ? `${s.checks_warned} with warnings` : '', s.checks_failed ? `${s.checks_failed} failed` : ''].filter(Boolean).join(', ') || 'all passed', ''],
  ].filter((item) => item[1] !== null);
  if (!items.length) return null;
  return h(
    'div',
    { class: 'tiles', testid: 'job-summary' },
    items.map(([label, value, sub, unit]) => {
      const item = tile(label);
      item.set(value, sub, unit);
      return item.el;
    }),
  );
}

function stoppedAt(job) {
  const p = job.progress || {};
  if (!FRAME_STAGES.includes(p.stage) || !isNumber(p.total_frames) || !isNumber(p.frame)) return null;
  return h('p', { class: 'muted num', testid: 'job-stopped-at' }, `Stopped in “${stageLabel(p.stage)}” at frame ${formatInt(p.frame)} of ${formatInt(p.total_frames)}.`);
}

function focusedJob(state) {
  const jobs = state.jobs || [];
  return jobById(state, state.focusJobId) || runningJob(state) || [...jobs].reverse().find((job) => job.status === 'queued') || jobs[0] || null;
}

function monitor(scope) {
  const title = h('span', { class: 'mon-title truncate' }, 'Job');
  const sub = h('span', { class: 'mon-sub' });
  const actions = h('div', { class: 'btn-row' });
  const body = h('div', { class: 'stack' });
  let shown = '';
  let panel = null;
  let panelScope = null;
  scope.add(() => {
    if (panelScope) panelScope.dispose();
  });

  function actionButtons(job) {
    const list = [];
    if (isDone(job) || job.report_url) {
      list.push(button({ label: isDone(job) ? 'Show results' : 'Show report', icon: 'eye', variant: isDone(job) ? 'primary' : '', onClick: () => showResults(job.id), testid: 'job-results' }));
    }
    if (isResumable(job)) list.push(button({ label: 'Resume', icon: 'render', variant: 'primary', onClick: () => resumeWithFeedback(job), testid: 'job-resume', tip: 'Queue the job again; mattes that were already computed are reused' }));
    if (isActive(job)) list.push(button({ label: 'Cancel', icon: 'stop', variant: 'danger', onClick: () => cancelWithConfirm(job), testid: 'job-cancel' }));
    list.push(button({ icon: 'terminal', tip: 'Show the log of this job', onClick: () => openLogDialog(job), testid: 'job-log-open' }));
    if (job.status !== 'running') list.push(button({ icon: 'trash', tip: 'Delete this job and its files', onClick: () => deleteWithConfirm(job), testid: 'job-delete' }));
    return list;
  }

  function render() {
    const state = store.get();
    const job = focusedJob(state);
    if (!job) {
      shown = '';
      if (panelScope) panelScope.dispose();
      panel = null;
      panelScope = null;
      title.textContent = 'Job';
      sub.textContent = '';
      fill(actions);
      fill(
        body,
        state.jobs === null
          ? state.jobsError
            ? errorNote(state.jobsError, { title: 'The job list could not be loaded', actions: [button({ label: 'Retry', icon: 'refresh', size: 'small', onClick: () => loadJobs() })] })
            : skeletonCards(1)
          : emptyState({ iconName: 'activity', title: 'No job yet', text: 'When you start a job, its progress and a live picture of the result appear here.', compact: true, plain: true, testid: 'monitor-empty' }),
      );
      return;
    }
    const key = `${job.id}:${job.status}:${job.report_url ? 1 : 0}:${state.models ? 1 : 0}`;
    if (key !== shown) {
      shown = key;
      if (panelScope) panelScope.dispose();
      panelScope = null;
      panel = null;
      title.textContent = job.name;
      title.setAttribute('data-tip', job.name);
      fill(sub, jobStatusBadge(job), h('span', { class: 'truncate' }, [jobSettingsLine(job, state), `added ${formatDateTime(job.created_at)}`].filter(Boolean).join(' · ')));
      fill(actions, actionButtons(job));
      body.dataset.jobId = job.id;
      body.dataset.status = job.status;

      if (job.status === 'running') {
        panelScope = createScope();
        panel = runningPanel(panelScope, () => focusedJob(store.get()));
        fill(body, panel.el);
      } else if (job.status === 'queued') {
        const active = (state.jobs || []).filter(isActive).length;
        fill(
          body,
          note({
            tone: 'info',
            iconName: 'clock',
            title: 'Waiting in the queue',
            text: active > 1 ? `${plural(active, 'job')} are queued or running. They are processed one at a time; this one starts by itself.` : 'The job starts as soon as the engine is free.',
            testid: 'job-queued',
          }),
        );
      } else if (isDone(job)) {
        const warnings = job.warnings || [];
        fill(
          body,
          note({
            tone: job.status === 'completed' ? 'ok' : 'warn',
            title: job.status === 'completed' ? 'Finished: every quality check passed' : 'Finished with warnings',
            text: `${job.output_name ? `${job.output_name} · ` : ''}${plural((job.outputs || []).length, 'file')} ready to download in Results.`,
            testid: 'job-finished',
          }),
          finishedSummary(job),
          warnings.length ? disclosure(plural(warnings.length, 'warning'), bullets(warnings, 'warn'), { open: warnings.length <= 3, testid: 'job-warnings-list' }) : null,
        );
      } else {
        const warnings = job.warnings || [];
        fill(
          body,
          jobErrorNote(job),
          stoppedAt(job),
          note({
            tone: 'info',
            text: 'Resume queues this job again with the same settings. Mattes that were already computed are reused, so only the missing frames go through the model.',
            testid: 'job-resume-info',
          }),
          finishedSummary(job),
          warnings.length ? disclosure(plural(warnings.length, 'warning'), bullets(warnings, 'warn'), { testid: 'job-warnings-list' }) : null,
        );
      }
    }
    if (panel) panel.update(job, state.streamMode);
  }

  scope.add(
    store.watch((state) => {
      const job = focusedJob(state);
      return [job, state.streamMode, state.jobs === null, state.jobsError, state.models, state.options];
    }, render),
  );

  return card({
    title: [title, sub],
    actions,
    body,
    testid: 'job-monitor',
    cls: 'monitor-card',
  });
}

// ------------------------------------------------------------------ queue and history

function rowDetail(job, live) {
  if (job.status === 'running') {
    const p = (live && live.id === job.id ? live.progress : job.progress) || {};
    return [stageLabel(p.stage || 'queued'), isNumber(p.overall_percent) ? formatPercent(p.overall_percent) : null].filter(Boolean).join(' · ');
  }
  if (job.status === 'queued') return 'Waiting in the queue';
  if (isDone(job)) {
    const s = job.summary || {};
    return [isNumber(s.frames) ? `${formatInt(s.frames)} frames` : null, jobDurationText(job) || null, (job.warnings || []).length ? plural(job.warnings.length, 'warning') : null].filter(Boolean).join(' · ');
  }
  return (job.error && job.error.message) || statusInfo(job.status).label;
}

function jobRow(job, state, focusedId) {
  const info = statusInfo(job.status);
  const live = state.liveJob;
  const actions = [];
  if (isDone(job)) actions.push(button({ icon: 'eye', size: 'small', variant: 'ghost', tip: 'Show results', testid: 'row-results', onClick: () => showResults(job.id) }));
  if (isResumable(job)) actions.push(button({ icon: 'render', size: 'small', variant: 'ghost', tip: 'Resume (stored mattes are reused)', testid: 'row-resume', onClick: () => resumeWithFeedback(job) }));
  if (isActive(job)) actions.push(button({ icon: 'x', size: 'small', variant: 'ghost', tip: job.status === 'running' ? 'Cancel this job' : 'Remove from the queue', testid: 'row-cancel', onClick: () => cancelWithConfirm(job) }));
  if (job.status !== 'running') actions.push(button({ icon: 'trash', size: 'small', variant: 'ghost', tip: 'Delete this job and its files', testid: 'row-delete', onClick: () => deleteWithConfirm(job) }));

  const running = job.status === 'running';
  const p = (running && live && live.id === job.id ? live.progress : job.progress) || {};
  const mini = running ? h('span', { class: 'job-row-bar' }, h('span')) : null;
  if (mini) setPercent(mini, isNumber(p.overall_percent) ? p.overall_percent : 0);

  return h(
    'li',
    { class: ['job-row', job.id === focusedId && 'is-focused'], testid: 'job-row', dataset: { jobId: job.id, status: job.status } },
    h(
      'button',
      { class: 'job-row-main', type: 'button', onClick: () => focusJob(job.id), 'aria-current': job.id === focusedId ? 'true' : null },
      running ? h('span', { class: 'pulse-dot' }) : h('span', { class: ['status-dot', info.tone] }),
      h(
        'span',
        { class: 'job-row-text' },
        h('span', { class: 'job-row-name truncate' }, job.name),
        h('span', { class: 'job-row-detail truncate', testid: 'job-row-status' }, h('span', { class: ['job-row-state', info.tone] }, info.label), ` · ${rowDetail(job, live)}`),
        h('span', { class: 'job-row-sub truncate' }, [jobSettingsLine(job, state), formatDateTime(job.created_at)].filter(Boolean).join(' · ')),
      ),
    ),
    h('div', { class: 'job-row-actions' }, actions),
    mini,
  );
}

function queuePanel(scope) {
  const list = h('ul', { class: 'job-list', testid: 'job-list' });
  const update = keyedList(list);
  const count = h('span', { class: 'card-sub', testid: 'job-count' });
  const body = h('div');

  scope.add(
    store.watch(
      (state) => {
        const live = state.liveJob;
        const focused = focusedJob(state);
        return [state.jobs, state.jobsError, focused ? focused.id : null, live ? `${live.id}:${Math.floor((live.progress || {}).overall_percent || 0)}:${(live.progress || {}).stage}` : null, state.models, state.options];
      },
      () => {
        const state = store.get();
        const jobs = state.jobs;
        if (!jobs) {
          count.textContent = '';
          fill(body, state.jobsError ? errorNote(state.jobsError, { title: 'The job list could not be loaded', actions: [button({ label: 'Retry', icon: 'refresh', size: 'small', onClick: () => loadJobs() })], testid: 'jobs-error' }) : skeletonCards(2));
          return;
        }
        const active = jobs.filter(isActive).length;
        count.textContent = jobs.length ? `${plural(jobs.length, 'job')}${active ? `, ${active} active` : ''}` : '';
        if (!jobs.length) {
          fill(body, emptyState({ iconName: 'list', title: 'No jobs', text: 'Queued, running and finished jobs are listed here.', compact: true, plain: true, testid: 'jobs-empty' }));
          return;
        }
        const focused = focusedJob(state);
        const focusedId = focused ? focused.id : null;
        const live = state.liveJob;
        update(
          jobs,
          (job) => job.id,
          (job) => {
            const { progress, ...rest } = job;
            const liveProgress = live && live.id === job.id ? live.progress || {} : progress || {};
            return JSON.stringify([rest, job.id === focusedId, job.status === 'running' ? [Math.floor(liveProgress.overall_percent || 0), liveProgress.stage] : null, !!state.models]);
          },
          (job) => jobRow(job, state, focusedId),
        );
        if (body.firstChild !== list) fill(body, list);
      },
    ),
  );

  return card({ title: ['Queue and history', count], body, flush: true, testid: 'job-queue' });
}

// ------------------------------------------------------------------ view

export function processView(scope) {
  return h(
    'div',
    { class: 'process', testid: 'view-process' },
    h(
      'div',
      { class: 'view-head' },
      h('div', { class: 'view-title' }, h('h1', null, 'Process'), h('p', null, 'Start the selected videos, follow the job that is running and manage the queue. Every figure shown here is measured by the engine.')),
      h(
        'div',
        { class: 'view-actions' },
        button({ label: 'Settings', icon: 'chevron-left', onClick: () => navigate('settings'), testid: 'process-back' }),
        button({ label: 'Results', iconRight: 'arrow-right', onClick: () => navigate('results'), testid: 'process-next' }),
      ),
    ),
    h('div', { class: 'process-layout' }, h('div', { class: 'stack' }, startPanel(scope), monitor(scope)), h('aside', { class: 'process-side' }, queuePanel(scope))),
  );
}
