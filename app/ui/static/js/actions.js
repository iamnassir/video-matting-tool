// Everything that changes application state: loading server data, selection, uploads, starting
// and controlling jobs, model downloads. Views call these functions and render from the store.

import { api, ApiError, toApiError } from './api.js';
import { store, config, selectedVideos, subjectFor, videoById } from './state.js';
import { defaultSubject, isDefaultSubject, parseValidationMessage, subjectPayload } from './config.js';
import { isActive, plural } from './format.js';
import { persist } from './persist.js';
import { toast, toastError } from './components/toast.js';

// ------------------------------------------------------------------ navigation

export function navigate(view, param = null) {
  const hash = param ? `#/${view}/${encodeURIComponent(param)}` : `#/${view}`;
  if (window.location.hash === hash) return;
  window.location.hash = hash;
}

export function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  store.set({ theme });
  persist.set('theme', theme);
}

// ------------------------------------------------------------------ loading server data

export async function loadSystem() {
  try {
    const system = await api.system();
    store.set({ system, systemError: null });
  } catch (error) {
    store.set({ systemError: toApiError(error) });
  }
}

export async function loadStats() {
  try {
    const stats = await api.stats();
    store.set({ stats, online: true });
  } catch (error) {
    const err = toApiError(error);
    if (err.isNetwork) store.set({ online: false });
  }
}

export async function loadOptions() {
  try {
    const options = await api.options();
    store.set({ options, optionsError: null });
    config.setBaseline(options.defaults);
  } catch (error) {
    store.set({ optionsError: toApiError(error) });
  }
}

export async function loadModels() {
  try {
    const models = await api.models();
    if (JSON.stringify(models) !== JSON.stringify(store.get().models)) store.set({ models });
    store.set({ modelsError: null });
  } catch (error) {
    store.set({ modelsError: toApiError(error) });
  }
}

export async function loadVideos() {
  try {
    const videos = await api.videos();
    applyVideos(videos);
    store.set({ videosError: null });
  } catch (error) {
    store.set({ videosError: toApiError(error) });
  }
}

function applyVideos(videos) {
  const state = store.get();
  const patch = {};
  if (JSON.stringify(videos) !== JSON.stringify(state.videos)) patch.videos = videos;
  const ids = new Set(videos.map((video) => video.id));
  const selectedIds = state.selectedIds.filter((id) => ids.has(id));
  if (selectedIds.length !== state.selectedIds.length) patch.selectedIds = selectedIds;
  if (state.activeId && !ids.has(state.activeId)) patch.activeId = selectedIds[0] || null;
  const subjectIds = Object.keys(state.subjects);
  if (subjectIds.some((id) => !ids.has(id))) {
    patch.subjects = Object.fromEntries(subjectIds.filter((id) => ids.has(id)).map((id) => [id, state.subjects[id]]));
  }
  store.set(patch);
}

/** Jobs without the fast-changing progress block: what the lists are built from. */
function jobShape(job) {
  const { progress, ...rest } = job;
  return rest;
}

export async function loadJobs() {
  try {
    const jobs = await api.jobs();
    applyJobs(jobs);
    store.set({ jobsError: null });
  } catch (error) {
    store.set({ jobsError: toApiError(error) });
  }
}

function applyJobs(jobs) {
  const state = store.get();
  const patch = {};
  const before = state.jobs ? JSON.stringify(state.jobs.map(jobShape)) : null;
  if (JSON.stringify(jobs.map(jobShape)) !== before) patch.jobs = jobs;
  const running = jobs.find((job) => job.status === 'running') || null;
  if (running) {
    // the stream (when attached) is fresher than the list; only fill in when there is none
    if (!state.liveJob || state.liveJob.id !== running.id || state.streamMode === null) patch.liveJob = running;
  } else if (state.liveJob && !jobs.some((job) => job.id === state.liveJob.id && isActive(job))) {
    patch.liveJob = null;
  }
  const ids = new Set(jobs.map((job) => job.id));
  if (state.focusJobId && !ids.has(state.focusJobId)) patch.focusJobId = null;
  if (state.resultJobId && !ids.has(state.resultJobId)) patch.resultJobId = null;
  store.set(patch);
}

/** Put one job (from an event or a single GET) into the list. */
export function mergeJob(job) {
  store.set((state) => {
    const jobs = state.jobs || [];
    const index = jobs.findIndex((item) => item.id === job.id);
    const patch = {};
    if (index < 0) patch.jobs = [job, ...jobs];
    else if (JSON.stringify(jobShape(jobs[index])) !== JSON.stringify(jobShape(job))) {
      patch.jobs = jobs.map((item) => (item.id === job.id ? job : item));
    }
    if (job.status === 'running') patch.liveJob = job;
    else if (state.liveJob && state.liveJob.id === job.id) patch.liveJob = null;
    return patch;
  });
}

// ------------------------------------------------------------------ selection

export function selectOnly(videoId) {
  store.set({ selectedIds: videoId ? [videoId] : [], activeId: videoId || null, startErrors: null });
}

export function toggleSelected(videoId) {
  store.set((state) => {
    const has = state.selectedIds.includes(videoId);
    const selectedIds = has ? state.selectedIds.filter((id) => id !== videoId) : [...state.selectedIds, videoId];
    let activeId = state.activeId;
    if (!has) activeId = videoId;
    else if (activeId === videoId) activeId = selectedIds[0] || null;
    return { selectedIds, activeId, startErrors: null };
  });
}

export function selectMany(videoIds) {
  store.set((state) => ({
    selectedIds: videoIds,
    activeId: videoIds.includes(state.activeId) ? state.activeId : videoIds[0] || null,
    startErrors: null,
  }));
}

export function setActive(videoId) {
  store.set({ activeId: videoId });
}

export function updateSubject(videoId, patch) {
  store.set((state) => ({
    subjects: { ...state.subjects, [videoId]: { ...subjectFor(state, videoId), ...patch } },
    startErrors: null,
  }));
}

export function resetSubject(videoId) {
  store.set((state) => ({ subjects: { ...state.subjects, [videoId]: defaultSubject(state.options) } }));
}

// ------------------------------------------------------------------ uploads

let uploadCounter = 0;
const uploadFiles = new Map(); // key -> File
const uploadAborts = new Map(); // key -> abort()
let pumping = false;
let batchIds = [];

function patchUpload(key, patch) {
  store.set((state) => ({ uploads: state.uploads.map((item) => (item.key === key ? { ...item, ...patch } : item)) }));
}

function removeUpload(key) {
  uploadFiles.delete(key);
  uploadAborts.delete(key);
  store.set((state) => ({ uploads: state.uploads.filter((item) => item.key !== key) }));
}

function extensionOf(name) {
  const dot = name.lastIndexOf('.');
  return dot >= 0 ? name.slice(dot).toLowerCase() : '';
}

/** Queue files for upload (one at a time, so several large files do not compete for the disk). */
export function queueUploads(files) {
  const { options } = store.get();
  const allowed = options ? options.input_extensions : null;
  const entries = [];
  for (const file of files) {
    uploadCounter += 1;
    const key = `upload-${uploadCounter}`;
    const entry = { key, name: file.name, size: file.size, loaded: 0, status: 'waiting', error: null };
    const ext = extensionOf(file.name);
    if (allowed && !allowed.includes(ext)) {
      entry.status = 'error';
      entry.error = new ApiError({
        code: 'unsupported_format',
        message: `'${file.name}' is not a supported video file (extension '${ext || 'none'}').`,
        hint: `Supported files: ${allowed.join(', ')}`,
      });
    } else if (file.size === 0) {
      entry.status = 'error';
      entry.error = new ApiError({ code: 'empty_file', message: `'${file.name}' is empty.` });
    } else {
      uploadFiles.set(key, file);
    }
    entries.push(entry);
  }
  if (!entries.length) return;
  store.set((state) => ({ uploads: [...state.uploads, ...entries] }));
  pumpUploads();
}

async function pumpUploads() {
  if (pumping) return;
  pumping = true;
  try {
    for (;;) {
      const next = store.get().uploads.find((item) => item.status === 'waiting');
      if (!next) break;
      const file = uploadFiles.get(next.key);
      if (!file) {
        removeUpload(next.key);
        continue;
      }
      patchUpload(next.key, { status: 'uploading' });
      const { promise, abort } = api.uploadVideo(file, (loaded, total) => {
        patchUpload(next.key, { loaded, size: total, status: loaded >= total ? 'analysing' : 'uploading' });
      });
      uploadAborts.set(next.key, abort);
      try {
        const video = await promise;
        removeUpload(next.key);
        batchIds.push(video.id);
        store.set((state) => ({ videos: [video, ...(state.videos || []).filter((item) => item.id !== video.id)] }));
      } catch (error) {
        const err = toApiError(error);
        if (err.isAborted) removeUpload(next.key);
        else patchUpload(next.key, { status: 'error', error: err });
        uploadFiles.delete(next.key);
        uploadAborts.delete(next.key);
      }
    }
  } finally {
    pumping = false;
  }
  if (batchIds.length) {
    const ids = batchIds;
    batchIds = [];
    selectMany(ids);
    toast({ type: 'ok', title: ids.length === 1 ? 'Video added' : `${ids.length} videos added`, timeout: 3000 });
    loadVideos();
  }
}

export function cancelUpload(key) {
  const abort = uploadAborts.get(key);
  if (abort) abort();
  else removeUpload(key);
}

export function dismissUpload(key) {
  removeUpload(key);
}

/** Register a file that already is on this computer. Throws ApiError (shown next to the path box). */
export async function importPath(path) {
  const video = await api.importVideo(path);
  store.set((state) => ({ videos: [video, ...(state.videos || []).filter((item) => item.id !== video.id)] }));
  selectOnly(video.id);
  return video;
}

export async function deleteVideo(videoId) {
  await api.deleteVideo(videoId);
  store.set((state) => {
    const selectedIds = state.selectedIds.filter((id) => id !== videoId);
    const subjects = { ...state.subjects };
    delete subjects[videoId];
    return {
      videos: (state.videos || []).filter((video) => video.id !== videoId),
      selectedIds,
      activeId: state.activeId === videoId ? selectedIds[0] || null : state.activeId,
      subjects,
    };
  });
  loadJobs();
}

// ------------------------------------------------------------------ starting jobs

/**
 * The requests a click on "Start" sends. One video -> POST /api/jobs. Several videos that share
 * the same subject selection -> one POST /api/jobs/batch. Several videos with individual subject
 * selections -> one POST /api/jobs per video (a click or a box only makes sense for its own video).
 */
export function buildStartRequests(state) {
  const videos = selectedVideos(state);
  if (!videos.length) return [];
  const base = config.payload();
  const subjectJson = videos.map((video) => {
    const subject = subjectFor(state, video.id);
    return isDefaultSubject(subject, state.options) ? '' : JSON.stringify(subjectPayload(subject, state.options));
  });
  const withSubject = (json) => (json ? { ...base, subject: JSON.parse(json) } : base);
  const shared = subjectJson.every((json) => json === subjectJson[0]);
  if (videos.length === 1 || shared) {
    return [{ videoIds: videos.map((video) => video.id), config: withSubject(subjectJson[0]) }];
  }
  return videos.map((video, index) => ({ videoIds: [video.id], config: withSubject(subjectJson[index]) }));
}

export async function startProcessing() {
  const state = store.get();
  if (state.starting) return [];
  const requests = buildStartRequests(state);
  if (!requests.length) return [];
  store.set({ starting: true, startErrors: null });
  const created = [];
  try {
    for (const request of requests) {
      if (request.videoIds.length === 1) created.push(await api.createJob(request.videoIds[0], request.config));
      else created.push(...(await api.createBatch(request.videoIds, request.config)));
    }
  } catch (error) {
    const err = toApiError(error);
    const parsed = err.status === 400 || err.status === 422 ? parseValidationMessage(err.message) : { fields: {}, general: [] };
    store.set({ startErrors: { ...parsed, error: err, queued: created.length } });
    toastError(err, created.length ? `Only ${plural(created.length, 'job')} could be queued` : 'The job could not be started');
  } finally {
    store.set({ starting: false });
  }
  if (created.length) {
    created.forEach(mergeJob);
    store.set({ focusJobId: created[0].id });
    toast({
      type: 'ok',
      title: created.length === 1 ? 'Job queued' : `${created.length} jobs queued`,
      text: created.length === 1 ? created[0].name : '',
      timeout: 3000,
    });
    loadJobs();
  }
  return created;
}

export function clearStartErrors() {
  if (store.get().startErrors) store.set({ startErrors: null });
}

// ------------------------------------------------------------------ job control

export async function cancelJob(jobId) {
  const job = await api.cancelJob(jobId);
  mergeJob(job);
  loadJobs();
  return job;
}

export async function resumeJob(jobId) {
  const job = await api.resumeJob(jobId);
  mergeJob(job);
  store.set({ focusJobId: job.id });
  loadJobs();
  return job;
}

export async function deleteJob(jobId) {
  await api.deleteJob(jobId);
  store.set((state) => ({
    jobs: (state.jobs || []).filter((job) => job.id !== jobId),
    liveJob: state.liveJob && state.liveJob.id === jobId ? null : state.liveJob,
    focusJobId: state.focusJobId === jobId ? null : state.focusJobId,
    resultJobId: state.resultJobId === jobId ? null : state.resultJobId,
  }));
  loadJobs();
}

/** Queue a new job that re-renders `jobId` with `changes` (sparse config relative to that job). */
export async function rerenderJob(jobId, changes) {
  const job = await api.rerenderJob(jobId, changes);
  mergeJob(job);
  store.set({ focusJobId: job.id });
  loadJobs();
  return job;
}

export function focusJob(jobId) {
  store.set({ focusJobId: jobId });
}

export function showResults(jobId) {
  store.set({ resultJobId: jobId });
  navigate('results', jobId);
}

// ------------------------------------------------------------------ model downloads

const downloadTimers = new Map();

function downloadKey(task) {
  return `${task.model}/${task.variant}`;
}

function setDownload(key, task) {
  store.set((state) => {
    const downloads = { ...state.downloads };
    if (task) downloads[key] = task;
    else delete downloads[key];
    return { downloads };
  });
  const running = {};
  for (const [k, t] of Object.entries(store.get().downloads)) if (t.status === 'running') running[k] = t.id;
  persist.set('downloadTasks', running);
}

function watchDownload(key, taskId) {
  if (downloadTimers.has(key)) clearInterval(downloadTimers.get(key));
  const stop = () => {
    clearInterval(downloadTimers.get(key));
    downloadTimers.delete(key);
  };
  const tick = async () => {
    try {
      const task = await api.task(taskId);
      setDownload(key, task);
      if (task.status === 'running') return;
      stop();
      await loadModels();
      if (task.status === 'completed') {
        setDownload(key, null);
        toast({ type: 'ok', title: 'Model weights downloaded', text: `${task.model} (${task.variant}) is ready to use.` });
      } else if (task.status === 'failed') {
        const err = task.error || {};
        toast({ type: 'err', title: 'The download failed', text: err.message || '', hint: err.hint || '' });
      }
    } catch (error) {
      const err = toApiError(error);
      if (err.status === 404) {
        // the server was restarted: the task is gone
        stop();
        setDownload(key, null);
        loadModels();
      }
    }
  };
  downloadTimers.set(key, setInterval(tick, 700));
  tick();
}

/** Start a download. Throws ApiError (for example 403 when the licence was not accepted). */
export async function startDownload(modelId, variantId, acceptLicense) {
  const task = await api.downloadModel(modelId, variantId, acceptLicense);
  const key = downloadKey(task);
  setDownload(key, task);
  watchDownload(key, task.id);
  return task;
}

export async function cancelDownload(key) {
  const task = store.get().downloads[key];
  if (!task) return;
  try {
    await api.cancelTask(task.id);
  } catch (error) {
    toastError(error, 'The download could not be cancelled');
  }
}

export function dismissDownload(key) {
  setDownload(key, null);
}

/** Re-attach to downloads that were running before the page was reloaded. */
export function resumeDownloadWatch() {
  const saved = persist.get('downloadTasks', {}) || {};
  for (const [key, taskId] of Object.entries(saved)) watchDownload(key, taskId);
}

// ------------------------------------------------------------------ helpers for views

export function videoName(state, videoId) {
  const video = videoById(state, videoId);
  return video ? video.filename : videoId;
}
