// Pieces shared by the Process and Results views: status badges, the description of a job's
// settings, error notes, the log dialog and the confirmed actions (cancel, resume, delete).

import { h, fill } from '../dom.js';
import { api, toApiError } from '../api.js';
import { store, modelById } from '../state.js';
import { describeSettingValue } from '../config.js';
import { formatSpan, isNumber, statusInfo } from '../format.js';
import { cancelJob, deleteJob, resumeJob } from '../actions.js';
import { badge, button, disclosure, errorNote, linkButton, loadingBlock, note } from './controls.js';
import { confirmDialog, openModal } from './modal.js';
import { toast, toastError } from './toast.js';

const STATUS_ICON = {
  queued: 'clock',
  running: 'activity',
  completed: 'check',
  completed_with_warnings: 'alert',
  failed: 'x',
  cancelled: 'minus',
  interrupted: 'alert',
};

export function jobStatusBadge(job) {
  const info = statusInfo(job.status);
  const el = badge(info.label, info.tone, STATUS_ICON[job.status] || null);
  el.setAttribute('data-testid', 'job-status');
  el.dataset.status = job.status;
  return el;
}

/** "Robust Video Matting · Balanced · green · automatic codec": the settings a job was created with. */
export function jobSettingsLine(job, state = store.get()) {
  const config = job.config || {};
  const model = modelById(state, config.model);
  const encoding = config.encoding || {};
  const parts = [
    model ? model.name : config.model,
    config.quality ? describeSettingValue('quality', config.quality, state.options) : null,
    config.background === 'custom' ? config.background_color : config.background,
    encoding.codec ? describeSettingValue('encoding.codec', encoding.codec, state.options) : null,
    config.max_frames ? `first ${config.max_frames} frames` : null,
  ];
  return parts.filter(Boolean).join(' · ');
}

/** Wall-clock time a job ran (null while it has not finished). */
export function jobDuration(job) {
  return isNumber(job.started_at) && isNumber(job.finished_at) ? Math.max(0, job.finished_at - job.started_at) : null;
}

export function jobDurationText(job) {
  const seconds = jobDuration(job);
  return seconds === null ? '' : formatSpan(seconds);
}

/**
 * The error of a failed / cancelled / interrupted job: message, hint and - when the API sent
 * them - technical details such as the last lines FFmpeg printed.
 */
export function jobErrorNote(job, { actions = null, testid = 'job-error' } = {}) {
  const error = job.error || null;
  const titles = { failed: 'The job failed', cancelled: 'The job was cancelled', interrupted: 'The job was interrupted' };
  if (!error) {
    return note({ tone: job.status === 'failed' ? 'err' : 'warn', title: titles[job.status] || 'The job did not finish', actions, testid });
  }
  const details = error.details && typeof error.details === 'object' ? error.details : {};
  const technical = [];
  for (const [key, value] of Object.entries(details)) {
    if (value === null || value === undefined || value === '') continue;
    technical.push(h('div', { class: 'section-title' }, key.replace(/_/g, ' ')), h('pre', { class: 'codeblock' }, typeof value === 'string' ? value : JSON.stringify(value, null, 2)));
  }
  const el = errorNote(error, {
    title: titles[job.status] || 'The job did not finish',
    tone: job.status === 'failed' ? 'err' : 'warn',
    actions,
    testid,
  });
  if (technical.length) {
    const body = el.querySelector('.note-body');
    const box = disclosure(`Technical details${error.code ? ` (${error.code})` : ''}`, technical, { plain: true, testid: 'job-error-details' });
    const actionsRow = body.querySelector('.note-actions');
    body.insertBefore(box, actionsRow);
  }
  return el;
}

// ------------------------------------------------------------------ log dialog

export function openLogDialog(job) {
  const pre = h('pre', { class: 'codeblock log', testid: 'job-log', tabIndex: 0 });
  const slot = h('div');
  const refresh = button({ label: 'Refresh', icon: 'refresh' });
  const load = async () => {
    fill(slot, loadingBlock('Loading the log'));
    try {
      const text = await api.log(job.id, 5000);
      if (text && text.trim()) {
        pre.textContent = text;
        fill(slot, pre);
        pre.scrollTop = pre.scrollHeight;
      } else {
        fill(slot, note({ tone: 'info', text: 'This job has not written a log yet.', testid: 'job-log-empty' }));
      }
    } catch (error) {
      fill(slot, errorNote(error, { title: 'The log could not be loaded' }));
    }
  };
  refresh.addEventListener('click', load);
  openModal({
    title: `Log · ${job.name}`,
    size: 'xwide',
    testid: 'log-dialog',
    body: slot,
    footer: [h('span', { class: 'grow' }, 'The last 5,000 lines.'), linkButton({ label: 'Open in a new tab', icon: 'external', href: api.logUrl(job.id), newTab: true }), refresh],
  });
  load();
}

// ------------------------------------------------------------------ confirmed actions

export async function cancelWithConfirm(job) {
  if (job.status === 'running') {
    const ok = await confirmDialog({
      title: 'Cancel this job?',
      message: `“${job.name}” stops after the batch that is being processed.`,
      detail: 'Partial output files are removed. The mattes computed so far are kept, so the job can be resumed later.',
      confirmLabel: 'Cancel the job',
      cancelLabel: 'Keep running',
      danger: true,
    });
    if (!ok) return;
  }
  try {
    await cancelJob(job.id);
  } catch (error) {
    toastError(error, 'The job could not be cancelled');
  }
}

export async function resumeWithFeedback(job) {
  try {
    await resumeJob(job.id);
    toast({ type: 'ok', title: 'Job queued again', text: job.name, timeout: 3000 });
  } catch (error) {
    toastError(error, 'The job could not be resumed');
  }
}

/** Ask, then delete the job with its files. Resolves to true when it was deleted. */
export async function deleteWithConfirm(job) {
  const dependants = (store.get().jobs || []).filter((item) => item.cache_job_id === job.id).length;
  const ok = await confirmDialog({
    title: 'Delete this job?',
    message: `“${job.name}”: the job, its output files, preview and report are deleted from the data folder.`,
    detail: dependants
      ? `${dependants === 1 ? 'A re-render uses' : `${dependants} re-renders use`} the stored mattes of this job; later re-renders of them will run the model again.`
      : 'The source video stays in the library.',
    confirmLabel: 'Delete',
    danger: true,
  });
  if (!ok) return false;
  try {
    await deleteJob(job.id);
    return true;
  } catch (error) {
    const err = toApiError(error);
    toastError(err, 'The job could not be deleted');
    return false;
  }
}
