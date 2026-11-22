// Application state: one observable store, the editable processing configuration and the
// selectors views use to read from them. Nothing in here talks to the network (see actions.js).

import { createStore } from './store.js';
import { ConfigModel, defaultSubject } from './config.js';

export const VIEWS = ['source', 'subject', 'settings', 'process', 'results', 'models'];

export const store = createStore({
  route: { view: 'source', param: null },
  theme: 'dark',

  // server data (null = not loaded yet)
  system: null,
  systemError: null,
  stats: null,
  online: true,
  options: null,
  optionsError: null,
  models: null,
  modelsError: null,
  videos: null,
  videosError: null,
  jobs: null,
  jobsError: null,

  // the job that is being processed right now, updated live (event stream or polling)
  liveJob: null,
  streamMode: null, // 'events' | 'polling' | null

  // user choices
  selectedIds: [], // selected source videos (several = batch)
  activeId: null, // the video shown in the preview and the subject editor
  subjects: {}, // subject selection per video id
  configVersion: 0, // increases whenever the processing configuration changes
  startErrors: null, // {fields, general, error} of the last rejected start request
  focusJobId: null, // job shown in the Process monitor
  resultJobId: null, // job shown in Results

  // transient
  uploads: [], // [{key, name, size, loaded, status, error}]
  downloads: {}, // model id -> download task
  starting: false,
});

/** The configuration for new jobs. Its baseline is /api/options "defaults". */
export const config = new ConfigModel();
config.subscribe(() => store.set((state) => ({ configVersion: state.configVersion + 1 })));

// ------------------------------------------------------------------ selectors

export function videoById(state, id) {
  return (state.videos || []).find((video) => video.id === id) || null;
}

export function selectedVideos(state) {
  return state.selectedIds.map((id) => videoById(state, id)).filter(Boolean);
}

export function activeVideo(state) {
  return videoById(state, state.activeId) || selectedVideos(state)[0] || null;
}

export function subjectFor(state, videoId) {
  return state.subjects[videoId] || defaultSubject(state.options);
}

export function modelById(state, id) {
  return (state.models || []).find((model) => model.id === id) || null;
}

/** The model the current configuration uses (null while models or options are not loaded). */
export function currentModel(state) {
  return modelById(state, config.get('model'));
}

export function jobById(state, id) {
  if (!id) return null;
  if (state.liveJob && state.liveJob.id === id) return state.liveJob;
  return (state.jobs || []).find((job) => job.id === id) || null;
}

export function runningJob(state) {
  if (state.liveJob && state.liveJob.status === 'running') return state.liveJob;
  return (state.jobs || []).find((job) => job.status === 'running') || null;
}

export function jobsOfVideo(state, videoId) {
  return (state.jobs || []).filter((job) => job.video_id === videoId);
}

export function anyModelReady(state) {
  return (state.models || []).some((model) => model.status && model.status.ready);
}

/** Variant a model uses for the chosen quality preset when the variant is left on automatic. */
export function automaticVariant(model, quality) {
  if (!model) return null;
  const id = (model.quality_variants && model.quality_variants[quality]) || model.default_variant;
  return model.variants.find((variant) => variant.id === id) || null;
}

export function variantReady(model, variantId) {
  const status = model && model.status && model.status.variants ? model.status.variants[variantId] : null;
  return !!(status && status.ready);
}

/** Licences (own and of dependencies) that are not plainly "allowed": they must be accepted before a download. */
export function restrictedLicenses(state, model) {
  const out = [];
  const seen = new Set();
  const visit = (m) => {
    if (!m || seen.has(m.id)) return;
    seen.add(m.id);
    if (m.license && m.license.commercial_use !== 'allowed') out.push({ id: m.id, name: m.name, license: m.license });
    for (const dep of m.depends_on || []) visit(modelById(state, dep));
  };
  visit(model);
  return out;
}
