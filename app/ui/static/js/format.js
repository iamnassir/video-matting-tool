// Formatting helpers for numbers, sizes, durations and API enums. Values that are missing
// (null / undefined / NaN) are rendered as an en dash instead of a made-up number.

export const DASH = '–';

const numberFormat = new Intl.NumberFormat('en-US');

export function isNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

export function formatInt(value) {
  return isNumber(value) ? numberFormat.format(Math.round(value)) : DASH;
}

export function formatFixed(value, digits = 1) {
  return isNumber(value) ? value.toFixed(digits) : DASH;
}

/** Trim a float to at most `digits` decimals without trailing zeros: 29.970 -> "29.97", 30 -> "30". */
export function formatTrim(value, digits = 3) {
  if (!isNumber(value)) return DASH;
  return String(parseFloat(value.toFixed(digits)));
}

export function formatBytes(bytes) {
  if (!isNumber(bytes)) return DASH;
  let value = bytes;
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  if (unit === 0) return `${Math.round(value)} B`;
  return `${value >= 100 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}

/** Megabytes as "512 MB" / "8.7 GB". */
export function formatMb(mb) {
  if (!isNumber(mb)) return DASH;
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${Math.round(mb)} MB`;
}

export function formatGbPair(usedMb, totalMb) {
  if (!isNumber(usedMb) || !isNumber(totalMb)) return DASH;
  return `${(usedMb / 1024).toFixed(1)} / ${(totalMb / 1024).toFixed(1)} GB`;
}

export function formatPercent(value, digits = 0) {
  return isNumber(value) ? `${value.toFixed(digits)}%` : DASH;
}

export function formatFps(value) {
  return isNumber(value) ? `${formatTrim(value, 3)} fps` : DASH;
}

/** Processing speed: one decimal below 100 frames per second. */
export function formatRate(value) {
  if (!isNumber(value)) return DASH;
  return value >= 100 ? value.toFixed(0) : value.toFixed(1);
}

/** Length of a clip: "0:12.4", "3:07", "1:02:45". */
export function formatDuration(seconds) {
  if (!isNumber(seconds) || seconds < 0) return DASH;
  const whole = Math.floor(seconds);
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const secs = whole % 60;
  if (hours > 0) return `${hours}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  if (seconds < 60) {
    const tenths = Math.floor((seconds - whole) * 10);
    return `0:${String(secs).padStart(2, '0')}.${tenths}`;
  }
  return `${minutes}:${String(secs).padStart(2, '0')}`;
}

/** Elapsed / remaining time in words: "45 s", "2 min 05 s", "1 h 12 min". */
export function formatSpan(seconds) {
  if (!isNumber(seconds) || seconds < 0) return DASH;
  const whole = Math.round(seconds);
  if (whole < 60) return seconds < 10 ? `${seconds.toFixed(1)} s` : `${whole} s`;
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const secs = whole % 60;
  if (hours > 0) return `${hours} h ${String(minutes).padStart(2, '0')} min`;
  return `${minutes} min ${String(secs).padStart(2, '0')} s`;
}

/** Player time code "mm:ss.mmm" (hours are added when needed). */
export function formatTimecode(seconds) {
  if (!isNumber(seconds) || seconds < 0) return '00:00.000';
  const ms = Math.round(seconds * 1000);
  const hours = Math.floor(ms / 3600000);
  const minutes = Math.floor((ms % 3600000) / 60000);
  const secs = Math.floor((ms % 60000) / 1000);
  const millis = ms % 1000;
  const base = `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${String(millis).padStart(3, '0')}`;
  return hours > 0 ? `${hours}:${base}` : base;
}

const dateTimeFormat = new Intl.DateTimeFormat(undefined, {
  day: 'numeric',
  month: 'short',
  hour: '2-digit',
  minute: '2-digit',
});

/** Unix time in seconds (as the API reports it) -> local short date and time. */
export function formatDateTime(epochSeconds) {
  if (!isNumber(epochSeconds)) return DASH;
  return dateTimeFormat.format(new Date(epochSeconds * 1000));
}

export function formatIsoDateTime(text) {
  if (!text) return DASH;
  const date = new Date(text);
  return Number.isNaN(date.getTime()) ? String(text) : dateTimeFormat.format(date);
}

export function capitalize(text) {
  const str = String(text ?? '');
  return str ? str.charAt(0).toUpperCase() + str.slice(1) : str;
}

/** "completed_with_warnings" -> "Completed with warnings". */
export function humanize(text) {
  return capitalize(String(text ?? '').replace(/[_-]+/g, ' '));
}

export function plural(count, singular, pluralForm) {
  return `${formatInt(count)} ${count === 1 ? singular : pluralForm || `${singular}s`}`;
}

/** Render any report value (number, string, boolean, list, object) as short readable text. */
export function formatValue(value) {
  if (value === null || value === undefined || value === '') return DASH;
  if (typeof value === 'boolean') return value ? 'yes' : 'no';
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return String(value);
    return Number.isInteger(value) ? String(value) : String(parseFloat(value.toPrecision(6)));
  }
  if (Array.isArray(value)) return value.map(formatValue).join(', ');
  if (typeof value === 'object') {
    return Object.entries(value)
      .map(([key, v]) => `${key}: ${formatValue(v)}`)
      .join(', ');
  }
  return String(value);
}

// ------------------------------------------------------------------ API vocabulary

const STAGE_LABELS = {
  queued: 'Waiting in the queue',
  probe: 'Analysing the input',
  subject: 'Identifying the subject',
  model_load: 'Loading the model',
  backtrack: 'Tracking back to the first frame',
  render: 'Matting and compositing',
  mux: 'Adding audio and metadata',
  export: 'Writing the deliverables',
  verify: 'Verifying the output',
  done: 'Finished',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

export function stageLabel(stage) {
  return STAGE_LABELS[stage] || humanize(stage || 'starting');
}

const STATUS = {
  queued: { label: 'Queued', tone: '' },
  running: { label: 'Running', tone: 'accent' },
  completed: { label: 'Completed', tone: 'ok' },
  completed_with_warnings: { label: 'Completed with warnings', tone: 'warn' },
  failed: { label: 'Failed', tone: 'err' },
  cancelled: { label: 'Cancelled', tone: '' },
  interrupted: { label: 'Interrupted', tone: 'warn' },
};

export function statusInfo(status) {
  return STATUS[status] || { label: humanize(status || 'unknown'), tone: '' };
}

export const ACTIVE_STATUSES = ['queued', 'running'];
export const DONE_STATUSES = ['completed', 'completed_with_warnings'];
export const RESUMABLE_STATUSES = ['failed', 'cancelled', 'interrupted'];

export function isActive(job) {
  return !!job && ACTIVE_STATUSES.includes(job.status);
}

export function isDone(job) {
  return !!job && DONE_STATUSES.includes(job.status);
}

export function isResumable(job) {
  return !!job && RESUMABLE_STATUSES.includes(job.status);
}
