// Keeps the store in step with the server: periodic polling of statistics and lists, and the live
// channel of the job that is being processed (Server-Sent Events, with polling as the fallback).

import { api, toApiError } from './api.js';
import { store } from './state.js';
import { isActive, isDone } from './format.js';
import { loadJobs, loadModels, loadOptions, loadStats, loadSystem, loadVideos, mergeJob, showResults } from './actions.js';
import { toast } from './components/toast.js';

const STATS_MS = 2000;
const JOBS_MS = 3000;
const VIDEOS_PENDING_MS = 2000;
const RETRY_MS = 8000;
const POLL_JOB_MS = 1000;

// ------------------------------------------------------------------ job stream

let stream = null; // {jobId, close}

function announce(job, previousStatus) {
  if (previousStatus !== 'running' && previousStatus !== 'queued') return;
  if (isDone(job)) {
    toast({
      type: job.status === 'completed' ? 'ok' : 'warn',
      title: job.status === 'completed' ? 'Processing finished' : 'Finished with warnings',
      text: job.name,
      action: { label: 'Show results', onClick: () => showResults(job.id) },
      timeout: 9000,
    });
  } else if (job.status === 'failed') {
    const error = job.error || {};
    toast({ type: 'err', title: `Processing failed: ${job.name}`, text: error.message || '', hint: error.hint || '' });
  } else if (job.status === 'cancelled') {
    toast({ type: 'info', title: 'Job cancelled', text: job.name, timeout: 4000 });
  }
}

function openJobStream(job) {
  const jobId = job.id;
  let closed = false;
  let source = null;
  let pollTimer = 0;
  let lastStatus = job.status;

  const apply = (next) => {
    if (closed || !next || next.id !== jobId) return;
    const previous = lastStatus;
    lastStatus = next.status;
    mergeJob(next);
    if (!isActive(next)) {
      close();
      announce(next, previous);
      loadJobs();
      loadStats();
    }
  };

  const startPolling = () => {
    if (closed || pollTimer) return;
    store.set({ streamMode: 'polling' });
    const tick = async () => {
      try {
        apply(await api.job(jobId));
      } catch (error) {
        if (toApiError(error).status === 404) {
          close();
          loadJobs();
        }
      }
    };
    pollTimer = setInterval(tick, POLL_JOB_MS);
    tick();
  };

  const close = () => {
    if (closed) return;
    closed = true;
    if (source) source.close();
    clearInterval(pollTimer);
    if (stream && stream.jobId === jobId) {
      stream = null;
      store.set({ streamMode: null });
    }
  };

  if (typeof window.EventSource === 'function') {
    try {
      source = new EventSource(job.events_url);
      source.onopen = () => {
        if (!closed && !pollTimer) store.set({ streamMode: 'events' });
      };
      source.onmessage = (event) => {
        let payload = null;
        try {
          payload = JSON.parse(event.data);
        } catch {
          return;
        }
        if (payload && payload.job) apply(payload.job);
      };
      source.onerror = () => {
        // the stream broke (server restart, proxy, unsupported): continue by polling the job
        if (source) source.close();
        source = null;
        startPolling();
      };
    } catch {
      startPolling();
    }
  } else {
    startPolling();
  }

  return { jobId, close };
}

/** Follow the job that is running (or, when none runs, the first queued one). */
function syncJobStream() {
  const { jobs } = store.get();
  const list = jobs || [];
  const target = list.find((job) => job.status === 'running') || [...list].reverse().find((job) => job.status === 'queued') || null;
  if (stream && (!target || stream.jobId !== target.id)) stream.close();
  if (target && !stream) stream = openJobStream(target);
}

// ------------------------------------------------------------------ polling

function every(ms, fn) {
  const tick = () => {
    if (!document.hidden) fn();
  };
  return setInterval(tick, ms);
}

export function startLiveUpdates() {
  store.watch((state) => state.jobs, syncJobStream);

  every(STATS_MS, loadStats);
  every(JOBS_MS, loadJobs);

  every(VIDEOS_PENDING_MS, () => {
    const { videos } = store.get();
    if ((videos || []).some((video) => video.proxy === 'pending')) loadVideos();
  });

  // endpoints that failed are retried quietly (engine modules or FFmpeg may become available)
  every(RETRY_MS, () => {
    const state = store.get();
    if (!state.online) return;
    if (state.optionsError || !state.options) loadOptions();
    if (state.systemError || !state.system) loadSystem();
    if (state.modelsError || !state.models) loadModels();
    if (state.videosError || !state.videos) loadVideos();
  });

  // when the server comes back after an outage everything is reloaded
  store.watch(
    (state) => state.online,
    (online, state) => {
      if (online && state.system !== undefined) refreshAll();
    },
    { immediate: false },
  );

  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
      loadStats();
      loadJobs();
    }
  });
}

export function refreshAll() {
  return Promise.all([loadSystem(), loadOptions(), loadModels(), loadVideos(), loadJobs(), loadStats()]);
}
