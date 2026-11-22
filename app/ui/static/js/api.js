// API client. Every request goes through `request()`, which turns any failure (network error,
// the API error envelope {"error": {code, message, hint, details}}, FastAPI validation errors,
// plain-text 500 pages) into one ApiError shape the interface can display.

export class ApiError extends Error {
  constructor({ status = 0, code = 'error', message, hint = null, details = null, url = '' }) {
    super(message || 'Request failed.');
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.hint = hint || null;
    this.details = details || null;
    this.url = url;
  }

  get isNetwork() {
    return this.status === 0 && this.code === 'network';
  }

  get isAborted() {
    return this.code === 'aborted';
  }
}

function fromFastApiDetail(detail) {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        const loc = Array.isArray(item.loc) ? item.loc.filter((part) => part !== 'body' && part !== 'query').join('.') : '';
        return loc ? `${loc}: ${item.msg}` : String(item.msg || '');
      })
      .filter(Boolean)
      .join('; ');
  }
  return '';
}

/** Build an ApiError from a failed response body (text) and its status code. */
export function errorFromBody(status, text, url) {
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = null;
  }
  if (data && typeof data === 'object') {
    if (data.error && typeof data.error === 'object') {
      return new ApiError({
        status,
        code: data.error.code || 'error',
        message: data.error.message || `The server answered with HTTP ${status}.`,
        hint: data.error.hint,
        details: data.error.details,
        url,
      });
    }
    if (data.detail !== undefined) {
      const message = fromFastApiDetail(data.detail);
      return new ApiError({
        status,
        code: status === 422 ? 'invalid_request' : 'error',
        message: message || `The server answered with HTTP ${status}.`,
        url,
      });
    }
  }
  if (status >= 500) {
    return new ApiError({
      status,
      code: 'server_error',
      message: `The server reported an internal error (HTTP ${status}).`,
      hint: 'Details are in the server console / log.',
      url,
    });
  }
  const short = (text || '').trim().slice(0, 300);
  return new ApiError({ status, code: 'error', message: short || `The server answered with HTTP ${status}.`, url });
}

function networkError(url) {
  return new ApiError({
    status: 0,
    code: 'network',
    message: 'The application server is not reachable.',
    hint: 'Check that the Video Matting Tool is still running, then try again.',
    url,
  });
}

function abortedError(url) {
  return new ApiError({ status: 0, code: 'aborted', message: 'The request was cancelled.', url });
}

/**
 * Turn anything into an ApiError (for uniform display): thrown errors, and the plain error objects
 * {code, message, hint, details} the API embeds in jobs and download tasks.
 */
export function toApiError(error) {
  if (error instanceof ApiError) return error;
  if (error && typeof error === 'object' && !(error instanceof Error) && typeof error.message === 'string') {
    return new ApiError({ code: error.code || 'error', message: error.message, hint: error.hint, details: error.details });
  }
  return new ApiError({ code: 'client_error', message: error && error.message ? error.message : String(error) });
}

async function request(method, url, { body, signal, as = 'json' } = {}) {
  const init = { method, signal, headers: {} };
  if (body !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(body);
  }
  if (as === 'json') init.headers.Accept = 'application/json';
  let response;
  try {
    response = await fetch(url, init);
  } catch (error) {
    if (error && error.name === 'AbortError') throw abortedError(url);
    throw networkError(url);
  }
  if (!response.ok) {
    let text = '';
    try {
      text = await response.text();
    } catch {
      text = '';
    }
    throw errorFromBody(response.status, text, url);
  }
  if (as === 'response') return response;
  if (response.status === 204) return null;
  try {
    if (as === 'text') return await response.text();
    if (as === 'blob') return await response.blob();
    return await response.json();
  } catch (error) {
    if (error && error.name === 'AbortError') throw abortedError(url);
    throw new ApiError({ status: response.status, code: 'bad_response', message: 'The server sent a response that could not be read.', url });
  }
}

const enc = encodeURIComponent;

export const api = {
  // ---------------------------------------------------------------- general
  system: (signal) => request('GET', '/api/system', { signal }),
  stats: (signal) => request('GET', '/api/system/stats', { signal }),
  options: (signal) => request('GET', '/api/options', { signal }),

  // ---------------------------------------------------------------- models
  models: (signal) => request('GET', '/api/models', { signal }),
  downloadModel: (modelId, variant, acceptLicense) =>
    request('POST', `/api/models/${enc(modelId)}/download`, { body: { variant: variant || null, accept_license: !!acceptLicense } }),
  task: (taskId) => request('GET', `/api/tasks/${enc(taskId)}`),
  cancelTask: (taskId) => request('POST', `/api/tasks/${enc(taskId)}/cancel`),

  // ---------------------------------------------------------------- videos
  videos: (signal) => request('GET', '/api/videos', { signal }),
  video: (videoId) => request('GET', `/api/videos/${enc(videoId)}`),
  deleteVideo: (videoId) => request('DELETE', `/api/videos/${enc(videoId)}`),
  importVideo: (path) => request('POST', '/api/videos/import', { body: { path } }),
  frameUrl: (videoId, index = 0, maxSide = 1280) => `/api/videos/${enc(videoId)}/frame?index=${index}&max_side=${maxSide}`,
  frame: (videoId, index, maxSide, signal) => request('GET', api.frameUrl(videoId, index, maxSide), { signal, as: 'blob' }),
  subjectPreview: (videoId, prompt, model, signal) =>
    request('POST', `/api/videos/${enc(videoId)}/subject-preview`, { body: { prompt, model: model || null }, signal }),

  /**
   * Upload one file as the raw request body. Returns {promise, abort}; `onProgress(loaded, total)`
   * reports the bytes that really left the browser.
   */
  uploadVideo(file, onProgress) {
    const url = `/api/videos/upload?filename=${enc(file.name)}`;
    const xhr = new XMLHttpRequest();
    const promise = new Promise((resolve, reject) => {
      xhr.open('POST', url);
      xhr.setRequestHeader('Content-Type', 'application/octet-stream');
      xhr.responseType = 'text';
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && onProgress) onProgress(event.loaded, event.total);
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch {
            reject(new ApiError({ status: xhr.status, code: 'bad_response', message: 'The server sent a response that could not be read.', url }));
          }
        } else {
          reject(errorFromBody(xhr.status, xhr.responseText, url));
        }
      };
      xhr.onerror = () => reject(networkError(url));
      xhr.onabort = () => reject(abortedError(url));
      xhr.send(file);
    });
    return { promise, abort: () => xhr.abort() };
  },

  // ---------------------------------------------------------------- jobs
  jobs: (signal) => request('GET', '/api/jobs', { signal }),
  job: (jobId, signal) => request('GET', `/api/jobs/${enc(jobId)}`, { signal }),
  createJob: (videoId, config) => request('POST', '/api/jobs', { body: { video_id: videoId, config } }),
  createBatch: (videoIds, config) => request('POST', '/api/jobs/batch', { body: { video_ids: videoIds, config } }),
  deleteJob: (jobId) => request('DELETE', `/api/jobs/${enc(jobId)}`),
  cancelJob: (jobId) => request('POST', `/api/jobs/${enc(jobId)}/cancel`),
  resumeJob: (jobId) => request('POST', `/api/jobs/${enc(jobId)}/resume`),
  rerenderJob: (jobId, config) => request('POST', `/api/jobs/${enc(jobId)}/rerender`, { body: { config } }),
  report: (jobId, signal) => request('GET', `/api/jobs/${enc(jobId)}/report`, { signal }),
  reportUrl: (jobId) => `/api/jobs/${enc(jobId)}/report`,
  reportTextUrl: (jobId) => `/api/jobs/${enc(jobId)}/report.txt`,
  logUrl: (jobId, tail = 20000) => `/api/jobs/${enc(jobId)}/log?tail=${tail}`,
  log: (jobId, tail = 2000, signal) => request('GET', api.logUrl(jobId, tail), { signal, as: 'text' }),

  /**
   * Latest live preview of a running job. Resolves to null while the server has no frame yet
   * (HTTP 204), otherwise to {blob, frameIndex, version}.
   */
  async live(job, signal) {
    const response = await request('GET', job.live_url, { signal, as: 'response' });
    if (response.status === 204) return null;
    const blob = await response.blob();
    const frame = Number.parseInt(response.headers.get('X-Frame-Index') || '', 10);
    const version = Number.parseInt(response.headers.get('X-Version') || '', 10);
    return {
      blob,
      frameIndex: Number.isFinite(frame) ? frame : null,
      version: Number.isFinite(version) ? version : null,
    };
  },
};
