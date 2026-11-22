// Results: the finished job - a frame-synchronous preview player over the 2 x 2 atlas the engine
// renders, every deliverable for download, the processing report as the engine wrote it, and the
// follow-up actions (re-render with other settings, delete).

import { h, fill, createScope } from '../dom.js';
import { icon } from '../icons.js';
import { api, toApiError } from '../api.js';
import { store, videoById, jobById } from '../state.js';
import { ConfigModel, parseValidationMessage, settingLabel } from '../config.js';
import {
  DASH,
  formatBytes,
  formatDateTime,
  formatInt,
  formatIsoDateTime,
  formatMb,
  formatRate,
  formatSpan,
  formatValue,
  humanize,
  isDone,
  isNumber,
  plural,
  statusInfo,
} from '../format.js';
import { navigate, rerenderJob, showResults, loadJobs } from '../actions.js';
import { badge, bullets, busyWhile, button, card, disclosure, emptyState, errorNote, kv, linkButton, loadingBlock, note, skeletonCards } from '../components/controls.js';
import { createPlayer } from '../components/player.js';
import { licenseBlock } from '../components/modelinfo.js';
import { openModal } from '../components/modal.js';
import { toast } from '../components/toast.js';
import { createSettingsForm } from '../components/settingsform.js';
import { deleteWithConfirm, jobDurationText, jobErrorNote, jobSettingsLine, jobStatusBadge, openLogDialog } from '../components/jobparts.js';

// reports do not change once a job has finished: keep them for the session (job id + finish time)
const reportCache = new Map();

function hasResult(job) {
  return isDone(job) || !!job.report_url || !!job.preview_url;
}

/** The job shown: the one named in the route / store, else the most recent finished one. */
function shownJob(state) {
  const jobs = state.jobs || [];
  const chosen = jobById(state, state.resultJobId);
  if (chosen) return chosen;
  return jobs.find(isDone) || jobs.find(hasResult) || null;
}

// ------------------------------------------------------------------ small pieces

const CHECK_LOOK = {
  pass: { label: 'Pass', tone: 'ok', icon: 'check' },
  warn: { label: 'Warning', tone: 'warn', icon: 'alert' },
  fail: { label: 'Failed', tone: 'err', icon: 'x' },
  skipped: { label: 'Skipped', tone: '', icon: 'minus' },
};

const OUTPUT_ICON = {
  main: 'film',
  alpha_matte: 'layers',
  png_sequence: 'image',
  prores_4444: 'film',
  green_screen: 'film',
  audio: 'music',
  matte_preview: 'grid',
};

function sectionTitle(text) {
  return h('div', { class: 'section-title' }, text);
}

function outputFacts(output) {
  return [
    output.is_folder ? 'folder, downloaded as .zip' : null,
    output.codec ? String(output.codec).toUpperCase() : null,
    output.container && output.container !== 'folder' ? `.${output.container}` : null,
    isNumber(output.frames) ? `${formatInt(output.frames)} frames` : null,
    isNumber(output.size_bytes) && output.size_bytes > 0 ? formatBytes(output.size_bytes) : null,
  ]
    .filter(Boolean)
    .join(' · ');
}

function downloadsCard(job) {
  const outputs = job.outputs || [];
  const rows = outputs.map((output) =>
    h(
      'li',
      { class: 'dl-row', testid: 'download-row', dataset: { kind: output.kind } },
      h('span', { class: 'dl-icon' }, icon(output.is_folder ? 'folder' : OUTPUT_ICON[output.kind] || 'film')),
      h(
        'span',
        { class: 'dl-text' },
        h('span', { class: 'dl-label' }, output.label || humanize(output.kind), output.kind === 'main' ? badge('Main', 'accent') : null),
        h('span', { class: 'dl-name truncate mono', tip: output.filename }, output.filename + (output.is_folder ? '/' : '')),
        h('span', { class: 'dl-facts num' }, outputFacts(output)),
      ),
      linkButton({
        label: output.is_folder ? 'Download .zip' : 'Download',
        icon: 'download',
        href: output.url,
        download: '',
        variant: output.kind === 'main' ? 'primary' : '',
        size: 'small',
        testid: `download-${output.kind}`,
      }),
    ),
  );
  const reportLinks = h(
    'div',
    { class: 'dl-extra' },
    job.report_url ? linkButton({ label: 'Report (JSON)', icon: 'report', href: api.reportUrl(job.id), newTab: true, size: 'small', variant: 'ghost', testid: 'report-json' }) : null,
    job.report_url ? linkButton({ label: 'Report (text)', icon: 'report', href: api.reportTextUrl(job.id), newTab: true, size: 'small', variant: 'ghost', testid: 'report-text' }) : null,
    linkButton({ label: 'Job log', icon: 'terminal', href: api.logUrl(job.id), newTab: true, size: 'small', variant: 'ghost', testid: 'job-log-link' }),
  );
  return card({
    title: 'Downloads',
    sub: outputs.length ? plural(outputs.length, 'file') : null,
    body: h(
      'div',
      { class: 'stack' },
      outputs.length
        ? h('ul', { class: 'dl-list', testid: 'downloads' }, rows)
        : note({ tone: 'info', text: 'This job produced no files to download.', testid: 'downloads-empty' }),
      reportLinks,
    ),
    testid: 'downloads-card',
  });
}

function previewCard(job, report, state) {
  if (!job.preview_url) {
    return card({
      title: 'Preview',
      body: note({ tone: 'info', text: 'This job has no preview video.', hint: isDone(job) ? '' : 'The preview is written while frames are rendered; the job stopped before that.', testid: 'preview-missing' }),
      testid: 'preview-card',
    });
  }
  const video = videoById(state, job.video_id);
  const input = (report && report.input) || (video && video.metadata) || {};
  const summary = job.summary || {};
  const player = createPlayer({
    src: job.preview_url,
    atlas: true,
    fps: isNumber(input.frame_rate_float) ? input.frame_rate_float : null,
    frameCount: isNumber(summary.frames) && summary.frames > 0 ? summary.frames : null,
    vfr: !!input.is_vfr,
    hasAudio: input.has_audio !== false,
    modeKey: 'resultsPlayerMode',
    testid: 'result-player',
  });
  const el = card({
    title: 'Preview',
    sub: 'original · subject · matte · composite, frame-synchronous',
    body: h('div', { class: 'stack' }, player.el, h('p', { class: 'field-hint' }, 'Keys: Space play / pause · , and . step one frame · ← → one second · 1-7 switch the view · F full screen. In Wipe, drag the divider.')),
    testid: 'preview-card',
  });
  return { el, player };
}

// ------------------------------------------------------------------ report

function summaryTiles(report) {
  const t = report.timing || {};
  const hw = report.hardware || {};
  const items = [
    ['Frames', formatInt(t.frames_processed), '', ''],
    ['Total time', formatSpan(t.total_seconds), isNumber(t.realtime_factor) ? `${formatRate(t.realtime_factor)}× real time` : '', ''],
    ['End to end', isNumber(t.average_fps) ? formatRate(t.average_fps) : DASH, 'render stage', 'fps'],
    ['Inference', isNumber(t.average_inference_fps) ? formatRate(t.average_inference_fps) : DASH, 'model only', 'fps'],
    ['From cache', formatInt(t.frames_from_cache), 'mattes reused', 'frames'],
    ['Device', hw.device || DASH, hw.device_name || '', ''],
  ];
  return h(
    'div',
    { class: 'tiles report-tiles', testid: 'report-tiles' },
    items.map(([label, value, sub, unit]) =>
      h(
        'div',
        { class: 'tile' },
        h('div', { class: 'tile-label' }, label),
        h('div', { class: 'tile-value', tip: sub || null }, value, value !== DASH && unit ? h('small', null, unit) : null),
        sub ? h('div', { class: 'tile-sub', tip: sub }, sub) : null,
      ),
    ),
  );
}

function matchCell(match) {
  if (match === true) return h('span', { class: 'match ok', tip: 'Identical / as expected' }, icon('check', 's'), 'match');
  if (match === false) return h('span', { class: 'match err', tip: 'Differs from the source' }, icon('x', 's'), 'differs');
  return h('span', { class: 'match faint', tip: 'Informational: not expected to be equal' }, 'info');
}

function comparisonTable(rows) {
  if (!rows || !rows.length) return null;
  return h(
    'div',
    { class: 'table-wrap' },
    h(
      'table',
      { class: 'table cmp-table', testid: 'report-comparison' },
      h('thead', null, h('tr', null, h('th', null, 'Property'), h('th', null, 'Input'), h('th', null, 'Output'), h('th', { class: 'shrink' }, 'Match'))),
      h(
        'tbody',
        null,
        rows.map((row) =>
          h(
            'tr',
            { class: row.match === false ? 'is-diff' : null, dataset: { match: String(row.match) } },
            h('td', null, row.property, row.note ? h('div', { class: 'cell-note' }, row.note) : null),
            h('td', { class: 'mono-num' }, formatValue(row.input)),
            h('td', { class: 'mono-num' }, formatValue(row.output)),
            h('td', { class: 'shrink' }, matchCell(row.match)),
          ),
        ),
      ),
    ),
  );
}

function checkList(checks) {
  if (!checks || !checks.length) return note({ tone: 'info', text: 'No quality checks were run for this job.' });
  const counts = {};
  for (const check of checks) counts[check.status] = (counts[check.status] || 0) + 1;
  const order = { fail: 0, warn: 1, pass: 2, skipped: 3 };
  const sorted = [...checks].sort((a, b) => (order[a.status] ?? 9) - (order[b.status] ?? 9));
  return h(
    'div',
    { class: 'stack-s' },
    h(
      'div',
      { class: 'check-counts' },
      ['pass', 'warn', 'fail', 'skipped'].filter((key) => counts[key]).map((key) => badge(`${counts[key]} ${CHECK_LOOK[key].label.toLowerCase()}`, CHECK_LOOK[key].tone, CHECK_LOOK[key].icon)),
    ),
    h(
      'ul',
      { class: 'qc-list', testid: 'report-checks' },
      sorted.map((check) => {
        const look = CHECK_LOOK[check.status] || { label: humanize(check.status), tone: '', icon: 'help' };
        const hasValues = (check.expected !== null && check.expected !== undefined) || (check.measured !== null && check.measured !== undefined);
        return h(
          'li',
          { class: ['qc-row', `is-${check.status}`], testid: 'report-check', dataset: { status: check.status, check: check.id } },
          h('span', { class: ['qc-icon', look.tone] }, icon(look.icon, 's')),
          h(
            'div',
            { class: 'qc-main' },
            h('div', { class: 'qc-head' }, h('span', { class: 'qc-name' }, check.name), h('span', { class: ['qc-state', look.tone] }, look.label)),
            check.detail ? h('div', { class: 'qc-detail break' }, check.detail) : null,
            hasValues
              ? h(
                  'div',
                  { class: 'qc-values num' },
                  check.expected !== null && check.expected !== undefined ? h('span', null, h('span', { class: 'faint' }, 'expected '), formatValue(check.expected)) : null,
                  check.measured !== null && check.measured !== undefined ? h('span', null, h('span', { class: 'faint' }, 'measured '), formatValue(check.measured)) : null,
                )
              : null,
          ),
        );
      }),
    ),
  );
}

function scalarPairs(object, prefix = '') {
  const pairs = [];
  for (const [key, value] of Object.entries(object || {})) {
    if (value === null || value === undefined) continue;
    if (Array.isArray(value)) {
      if (!value.length) continue;
      if (value.every((item) => item === null || typeof item !== 'object')) pairs.push([`${prefix}${humanize(key)}`, formatValue(value)]);
      else pairs.push([`${prefix}${humanize(key)}`, plural(value.length, 'entry', 'entries')]);
    } else if (typeof value === 'object') {
      pairs.push(...scalarPairs(value, `${prefix}${humanize(key)} · `));
    } else {
      pairs.push([`${prefix}${humanize(key)}`, formatValue(value)]);
    }
  }
  return pairs;
}

function modelBlock(report) {
  const m = report.model || {};
  if (!m.id) return null;
  const runtime = m.runtime && typeof m.runtime === 'object' ? m.runtime : {};
  return h(
    'div',
    { class: 'stack-s', testid: 'report-model' },
    kv([
      ['Model', m.name ? `${m.name} (${m.id})` : m.id],
      ['Variant', m.variant || null],
      ['Output', m.alpha_type === 'alpha' ? 'True alpha matte' : m.alpha_type ? `${humanize(m.alpha_type)} (refined to alpha)` : null],
      ['Subject domain', m.subject_domain ? humanize(m.subject_domain) : null],
      ...Object.entries(runtime)
        .filter(([key]) => !['model', 'variant', 'options'].includes(key))
        .filter(([, value]) => value !== null && value !== undefined && typeof value !== 'object')
        .map(([key, value]) => [humanize(key), formatValue(value)]),
    ]),
    m.license ? licenseBlock(m) : null,
    m.limitations && m.limitations.length ? disclosure(`Known limitations (${m.limitations.length})`, bullets(m.limitations), { plain: true }) : null,
  );
}

function encodingBlock(report) {
  const e = report.encoding || {};
  if (!Object.keys(e).length) return null;
  const args = Array.isArray(e.video_args) ? e.video_args.join(' ') : null;
  const mux = Array.isArray(e.mux_command) ? e.mux_command.join(' ') : null;
  const known = new Set(['codec', 'container', 'encoder', 'pix_fmt', 'lossless', 'background', 'audio', 'video_args', 'mux_command']);
  return h(
    'div',
    { class: 'stack-s', testid: 'report-encoding' },
    kv([
      ['Codec', e.codec ? `${String(e.codec).toUpperCase()}${e.encoder ? ` (${e.encoder})` : ''}` : null],
      ['Container', e.container ? `.${e.container}` : null],
      ['Pixel format', e.pix_fmt || null],
      ['Lossless', typeof e.lossless === 'boolean' ? (e.lossless ? 'yes' : 'no') : null],
      ['Background', e.background ? h('span', { class: 'summary-swatch' }, /^#[0-9a-f]{6}$/i.test(e.background) ? h('span', { class: 'swatch', style: { '--c': e.background } }) : null, h('span', { class: 'mono' }, String(e.background))) : null],
      ['Audio', e.audio ? String(e.audio) : null],
      ...Object.entries(e)
        .filter(([key, value]) => !known.has(key) && value !== null && value !== undefined && typeof value !== 'object')
        .map(([key, value]) => [humanize(key), formatValue(value)]),
    ]),
    args || mux
      ? disclosure(
          'FFmpeg arguments',
          [args ? [sectionTitle('Video encoder'), h('pre', { class: 'codeblock' }, args)] : null, mux ? [sectionTitle('Final mux'), h('pre', { class: 'codeblock' }, mux)] : null],
          { plain: true },
        )
      : null,
  );
}

function hardwareBlock(report) {
  const hw = report.hardware || {};
  return kv(
    [
      ['Device', hw.device ? `${hw.device}${hw.device_name ? ` (${hw.device_name})` : ''}` : null],
      ['Requested', hw.requested_device || null],
      ['Fallback', hw.fallback_reason || null],
      ['Precision', hw.precision || null],
      ['Batch size', isNumber(hw.final_batch_size) ? `${hw.final_batch_size}${hw.batch_reductions ? ` (reduced ${plural(hw.batch_reductions, 'time')} after running out of memory)` : ''}` : null],
      ['VRAM', isNumber(hw.vram_total_mb) ? `${formatMb(hw.vram_total_mb)} total${isNumber(hw.vram_free_at_start_mb) ? `, ${formatMb(hw.vram_free_at_start_mb)} free at start` : ''}${isNumber(hw.vram_peak_allocated_mb) ? `, peak ${formatMb(hw.vram_peak_allocated_mb)}` : ''}` : null],
      ['RAM', isNumber(hw.ram_total_mb) ? `${formatMb(hw.ram_total_mb)} installed${isNumber(hw.ram_peak_process_mb) ? `, peak of the engine ${formatMb(hw.ram_peak_process_mb)}` : ''}` : null],
      ['Processor', hw.cpu ? `${hw.cpu}${hw.cpu_threads ? ` (${hw.cpu_threads} threads)` : ''}` : null],
      ['System', hw.os || null],
      ['Python', hw.python || null],
      ['PyTorch', hw.torch || null],
      ['CUDA', hw.cuda || null],
      ['FFmpeg', hw.ffmpeg || null],
    ],
    'report-hardware',
  );
}

function timingBlock(report) {
  const stages = (report.timing && report.timing.stages) || {};
  const entries = Object.entries(stages).filter(([, seconds]) => isNumber(seconds));
  if (!entries.length) return null;
  const total = entries.reduce((sum, [, seconds]) => sum + seconds, 0) || 1;
  return h(
    'ul',
    { class: 'stage-times', testid: 'report-stages' },
    entries.map(([name, seconds]) =>
      h(
        'li',
        null,
        h('span', { class: 'stage-name' }, humanize(name)),
        h('span', { class: 'stage-bar', style: { '--p': ((seconds / total) * 100).toFixed(2) } }, h('span')),
        h('span', { class: 'stage-secs num' }, formatSpan(seconds)),
      ),
    ),
  );
}

function subjectBlock(report) {
  const s = report.subject || {};
  if (!Object.keys(s).length) return null;
  return kv([
    ['Selection', s.source ? String(s.source) : null],
    ['Reference frame', isNumber(s.frame_index) ? formatInt(s.frame_index) : null],
    ['Label', s.label || null],
    ['Text', s.text || null],
    ['Score', isNumber(s.score) ? s.score.toFixed(3) : null],
    ['Box', Array.isArray(s.box) ? s.box.map((v) => Math.round(v)).join(', ') : null],
    ['Clicks', Array.isArray(s.points) && s.points.length ? String(s.points.length) : null],
    ['Notes', Array.isArray(s.notes) && s.notes.length ? s.notes.join(' ') : null],
  ]);
}

function reportCard(job, report, reportError) {
  if (reportError) {
    return card({
      title: 'Processing report',
      body: reportError.status === 404 ? note({ tone: 'info', text: 'This job has no report.', testid: 'report-missing' }) : errorNote(reportError, { title: 'The report could not be loaded', testid: 'report-error' }),
      testid: 'report-card',
    });
  }
  if (!report) return card({ title: 'Processing report', body: loadingBlock('Loading the report'), testid: 'report-card' });

  const info = statusInfo(report.status);
  const checks = report.checks || [];
  const failed = checks.filter((check) => check.status === 'fail').length;
  const warned = checks.filter((check) => check.status === 'warn').length;
  const statusText =
    report.status === 'completed'
      ? 'Every quality check passed and the output matches the source.'
      : report.status === 'completed_with_warnings'
        ? [failed ? `${plural(failed, 'check')} failed` : null, warned ? `${plural(warned, 'check')} with warnings` : null, (report.warnings || []).length ? plural(report.warnings.length, 'warning') : null].filter(Boolean).join(', ') + '. Details below.'
        : '';
  const matte = report.matte_stats && Object.keys(report.matte_stats).length ? scalarPairs(report.matte_stats) : [];

  const body = h(
    'div',
    { class: 'report', testid: 'report' },
    h(
      'div',
      { class: 'report-status', dataset: { status: report.status } },
      badge(info.label, info.tone, report.status === 'completed' ? 'check' : report.status === 'completed_with_warnings' ? 'alert' : 'x'),
      statusText ? h('span', { class: 'muted' }, statusText) : null,
      h('span', { class: 'grow' }),
      h('span', { class: 'faint num' }, [report.resumed ? 'resumed run' : null, `written ${formatIsoDateTime(report.created_at)}`, report.app_version ? `v${report.app_version}` : null].filter(Boolean).join(' · ')),
    ),
    report.error ? errorNote(report.error, { title: 'The run reported an error', testid: 'report-run-error' }) : null,
    summaryTiles(report),
    (report.warnings || []).length
      ? h('div', { class: 'report-section', testid: 'report-warnings' }, sectionTitle(`Warnings (${report.warnings.length})`), bullets(report.warnings, 'warn'))
      : null,
    h('div', { class: 'report-section' }, sectionTitle('Input and output'), comparisonTable(report.comparison) || h('p', { class: 'muted' }, 'No comparison available.')),
    h('div', { class: 'report-section' }, sectionTitle('Quality verification'), checkList(checks)),
    h(
      'div',
      { class: 'report-cols' },
      h('div', { class: 'report-section' }, sectionTitle('Model and licence'), modelBlock(report) || h('p', { class: 'muted' }, DASH)),
      h('div', { class: 'report-section' }, sectionTitle('Encoding'), encodingBlock(report) || h('p', { class: 'muted' }, DASH)),
    ),
    h(
      'div',
      { class: 'report-cols' },
      h('div', { class: 'report-section' }, sectionTitle('Hardware'), hardwareBlock(report)),
      h('div', { class: 'report-section' }, sectionTitle('Time per stage'), timingBlock(report) || h('p', { class: 'muted' }, DASH), subjectBlock(report) ? [sectionTitle('Subject'), subjectBlock(report)] : null),
    ),
    matte.length ? disclosure('Matte statistics', kv(matte, 'report-matte'), { testid: 'report-matte-box' }) : null,
    h('div', { class: 'report-paths' }, kv([['Input file', report.input_path || null], ['Output file', report.output_path || null]])),
  );
  return card({ title: 'Processing report', body, testid: 'report-card' });
}

// ------------------------------------------------------------------ re-render dialog

function openRerenderDialog(job) {
  const scope = createScope();
  const model = new ConfigModel(JSON.parse(JSON.stringify(job.config || {})));
  const form = createSettingsForm({ model, scope, rerender: true });
  const impact = h('div', { testid: 'rerender-impact' });
  const errorSlot = h('div');
  const submit = button({ label: 'Queue re-render', icon: 'render', variant: 'primary', testid: 'rerender-submit' });

  function sync() {
    const changes = model.changedPaths();
    const fingerprint = form.fingerprintChanges();
    submit.disabled = changes.length === 0;
    if (!changes.length) {
      fill(impact, note({ tone: 'info', text: 'Change a setting below - for example the background - to create a new version of this result.' }));
    } else if (fingerprint.length) {
      fill(impact, note({ tone: 'warn', title: 'The model runs again', text: `You changed the ${fingerprint.map((item) => item.label).join(', ')}: the raw mattes depend on it, so every frame goes through the model again.` }));
    } else {
      fill(impact, note({ tone: 'ok', title: 'Stored mattes are reused', text: `Only ${plural(changes.length, 'setting')} of refinement, background or output changed: the model does not run again, so the re-render takes a fraction of the original time.` }));
    }
  }
  scope.add(model.subscribe(() => {
    sync();
    fill(errorSlot);
    form.setErrors({});
  }));
  sync();

  const modal = openModal({
    title: `Re-render · ${job.name}`,
    size: 'xwide',
    testid: 'rerender-dialog',
    body: h(
      'div',
      { class: 'stack' },
      h('p', { class: 'muted' }, 'A new job is created for the same video with the settings of this job plus your changes. Mattes stored by this job are reused when only the matte refinement, the background, the encoding or the extra files change; the model, variant, quality preset, precision or frame limit make it run again.'),
      impact,
      errorSlot,
      form.el,
    ),
    footer: [h('span', { class: 'grow' }), button({ label: 'Cancel', onClick: () => modal.close(), testid: 'rerender-cancel' }), submit],
    onClose: () => scope.dispose(),
  });

  submit.addEventListener('click', () =>
    busyWhile(submit, async () => {
      try {
        const created = await rerenderJob(job.id, model.payload());
        modal.close();
        toast({ type: 'ok', title: 'Re-render queued', text: created.name, timeout: 3500 });
        navigate('process', created.id);
      } catch (error) {
        const err = toApiError(error);
        const parsed = err.status === 400 || err.status === 422 ? parseValidationMessage(err.message) : { fields: {}, general: [] };
        const unplaced = form.setErrors(parsed.fields);
        const lines = [...parsed.general, ...unplaced.map((item) => `${settingLabel(item.path)}: ${item.text}`)];
        fill(
          errorSlot,
          Object.keys(parsed.fields).length || parsed.general.length
            ? note({ tone: 'err', title: 'The server did not accept these settings', children: lines.length ? bullets(lines) : null, hint: err.hint || '', testid: 'rerender-error' })
            : errorNote(err, { title: 'The re-render could not be queued', testid: 'rerender-error' }),
        );
        errorSlot.scrollIntoView({ block: 'nearest' });
      }
    }),
  );
}

// ------------------------------------------------------------------ side list

function resultList(scope) {
  const list = h('ul', { class: 'job-list', testid: 'result-list' });
  const body = h('div');
  const count = h('span', { class: 'card-sub' });
  scope.add(
    store.watch(
      (state) => {
        const shown = shownJob(state);
        return [state.jobs, shown ? shown.id : null, state.models];
      },
      ([jobs, shownId]) => {
        const items = (jobs || []).filter(hasResult);
        count.textContent = items.length ? String(items.length) : '';
        if (!jobs) {
          fill(body, skeletonCards(2));
          return;
        }
        if (!items.length) {
          fill(body, emptyState({ iconName: 'list', title: 'No results yet', text: 'Finished jobs appear here.', compact: true, plain: true }));
          return;
        }
        fill(
          list,
          items.map((job) => {
            const info = statusInfo(job.status);
            const s = job.summary || {};
            return h(
              'li',
              { class: ['job-row', job.id === shownId && 'is-focused'], testid: 'result-row', dataset: { jobId: job.id, status: job.status } },
              h(
                'button',
                { class: 'job-row-main', type: 'button', onClick: () => showResults(job.id), 'aria-current': job.id === shownId ? 'true' : null },
                h('span', { class: ['status-dot', info.tone] }),
                h(
                  'span',
                  { class: 'job-row-text' },
                  h('span', { class: 'job-row-name truncate' }, job.name),
                  h('span', { class: 'job-row-detail truncate' }, h('span', { class: ['job-row-state', info.tone] }, info.label), isNumber(s.frames) ? ` · ${formatInt(s.frames)} frames` : '', jobDurationText(job) ? ` · ${jobDurationText(job)}` : ''),
                  h('span', { class: 'job-row-sub truncate' }, [jobSettingsLine(job), formatDateTime(job.finished_at || job.created_at)].filter(Boolean).join(' · ')),
                ),
              ),
            );
          }),
        );
        if (body.firstChild !== list) fill(body, list);
      },
    ),
  );
  return card({ title: ['Results', count], body, flush: true, testid: 'results-list-card' });
}

// ------------------------------------------------------------------ main panel

function resultPanel(scope, sideSlot) {
  const el = h('div', { class: 'stack', testid: 'result-panel' });
  let shownKey = '';
  let panelScope = null;
  let player = null;
  const release = () => {
    if (player) player.destroy();
    player = null;
    if (panelScope) panelScope.dispose();
    panelScope = null;
  };
  scope.add(release);

  async function fetchReport(job, key) {
    if (reportCache.has(key)) return;
    try {
      const report = await api.report(job.id);
      reportCache.set(key, { report, error: null });
    } catch (error) {
      reportCache.set(key, { report: null, error: toApiError(error) });
    }
    if (!scope.disposed && shownKey.startsWith(key)) render(true);
  }

  function header(job, state) {
    const actions = [
      button({ label: 'Re-render', icon: 'render', onClick: () => openRerenderDialog(job), testid: 'rerender-open', tip: 'Create a new version with other settings', disabled: !isDone(job) && !job.report_url }),
      button({ icon: 'terminal', tip: 'Show the log of this job', onClick: () => openLogDialog(job), testid: 'result-log' }),
      button({
        icon: 'trash',
        tip: 'Delete this job and its files',
        testid: 'result-delete',
        onClick: async () => {
          if (await deleteWithConfirm(job)) toast({ type: 'info', title: 'Job deleted', text: job.name, timeout: 3000 });
        },
      }),
    ];
    const source = job.cache_job_id ? jobById(state, job.cache_job_id) : null;
    return h(
      'section',
      { class: 'card result-head', testid: 'result-head' },
      h(
        'div',
        { class: 'result-head-main' },
        h('h2', { class: 'truncate', tip: job.name }, job.name),
        h(
          'div',
          { class: 'mon-sub' },
          jobStatusBadge(job),
          h('span', { class: 'break' }, [jobSettingsLine(job, state), job.finished_at ? `finished ${formatDateTime(job.finished_at)}` : null, jobDurationText(job) ? `took ${jobDurationText(job)}` : null].filter(Boolean).join(' · ')),
        ),
        job.cache_job_id ? h('div', { class: 'field-hint' }, `Re-render of ${source ? `“${source.name}” from ${formatDateTime(source.created_at)}` : `job ${job.cache_job_id}`}: its stored mattes were used where the settings allowed.`) : null,
      ),
      h('div', { class: 'btn-row' }, actions),
    );
  }

  function render(force = false) {
    const state = store.get();
    const job = shownJob(state);
    if (!job) {
      shownKey = '';
      release();
      fill(sideSlot);
      fill(
        el,
        state.jobs === null
          ? state.jobsError
            ? errorNote(state.jobsError, { title: 'The job list could not be loaded', actions: [button({ label: 'Retry', icon: 'refresh', size: 'small', onClick: () => loadJobs() })] })
            : skeletonCards(2)
          : card({
              body: emptyState({
                iconName: 'eye',
                title: 'No result yet',
                text: 'When a job has finished, its preview, the files to download and the processing report appear here.',
                actions: [button({ label: 'Open Process', iconRight: 'arrow-right', variant: 'primary', onClick: () => navigate('process') })],
                testid: 'results-empty',
              }),
            }),
      );
      return;
    }
    const reportKey = `${job.id}:${job.finished_at || ''}`;
    const key = `${reportKey}:${job.status}:${job.preview_url ? 1 : 0}:${state.models ? 1 : 0}`;
    if (key === shownKey && !force) return;
    const cached = reportCache.get(reportKey);
    // the player is kept when only the report arrived (it would restart otherwise)
    const keepPlayer = force && player && shownKey === key;
    shownKey = key;

    if (!isDone(job) && !job.report_url && !job.preview_url) {
      release();
      fill(sideSlot);
      fill(
        el,
        header(job, state),
        job.status === 'queued' || job.status === 'running'
          ? note({ tone: 'info', title: 'This job has not finished yet', text: 'Its results appear here when it is done.', actions: [button({ label: 'Follow it in Process', size: 'small', iconRight: 'arrow-right', onClick: () => navigate('process', job.id) })], testid: 'result-pending' })
          : jobErrorNote(job, { actions: [button({ label: 'Open in Process', size: 'small', iconRight: 'arrow-right', onClick: () => navigate('process', job.id) })] }),
      );
      return;
    }

    if (job.report_url && !cached) fetchReport(job, reportKey);
    let previewEl;
    if (keepPlayer) {
      previewEl = el.querySelector('[data-testid="preview-card"]');
    } else {
      release();
      panelScope = createScope();
      const preview = previewCard(job, cached ? cached.report : null, state);
      if (preview.player) player = preview.player;
      previewEl = preview.el || preview;
    }
    fill(sideSlot, downloadsCard(job));
    fill(
      el,
      header(job, state),
      isDone(job) ? null : jobErrorNote(job),
      previewEl,
      job.report_url ? reportCard(job, cached ? cached.report : null, cached ? cached.error : null) : null,
    );
  }

  scope.add(
    store.watch((state) => {
      const job = shownJob(state);
      return [job, state.jobs === null, state.jobsError, state.models];
    }, () => render()),
  );
  return el;
}

// ------------------------------------------------------------------ view

export function resultsView(scope) {
  const side = h('div', { class: 'results-downloads' });
  return h(
    'div',
    { class: 'results', testid: 'view-results' },
    h(
      'div',
      { class: 'view-head' },
      h('div', { class: 'view-title' }, h('h1', null, 'Results'), h('p', null, 'Compare the result with the original, download the files and read how the output was verified.')),
      h('div', { class: 'view-actions' }, button({ label: 'Process', icon: 'chevron-left', onClick: () => navigate('process'), testid: 'results-back' })),
    ),
    h('div', { class: 'results-layout' }, resultPanel(scope, side), h('aside', { class: 'results-side stack' }, side, resultList(scope))),
  );
}
