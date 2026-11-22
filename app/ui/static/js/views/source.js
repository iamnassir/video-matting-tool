// Source: upload or register videos, the library of registered videos and the original preview.

import { h, fill, keyedList } from '../dom.js';
import { icon } from '../icons.js';
import { api } from '../api.js';
import { store, activeVideo, jobsOfVideo } from '../state.js';
import {
  formatBytes,
  formatDateTime,
  formatDuration,
  formatFps,
  formatInt,
  formatTrim,
  isNumber,
  plural,
} from '../format.js';
import {
  cancelUpload,
  deleteVideo,
  dismissUpload,
  importPath,
  loadVideos,
  navigate,
  queueUploads,
  selectMany,
  selectOnly,
  setActive,
  toggleSelected,
} from '../actions.js';
import { badge, bullets, busyWhile, button, card, disclosure, emptyState, errorNote, kv, note, progressBar, skeletonCards } from '../components/controls.js';
import { confirmDialog } from '../components/modal.js';
import { toastError } from '../components/toast.js';
import { createPlayer } from '../components/player.js';

// ------------------------------------------------------------------ metadata helpers

export function videoFacts(video) {
  const meta = video.metadata || {};
  const v = meta.video || {};
  const color = v.color || {};
  const hdrKind = color.transfer === 'smpte2084' ? 'HDR PQ' : color.transfer === 'arib-std-b67' ? 'HDR HLG' : null;
  const hasHdrMeta = v.hdr && Object.keys(v.hdr).length > 0;
  const duration = isNumber(meta.video_duration) ? meta.video_duration : isNumber(meta.duration) ? meta.duration : null;
  return {
    meta,
    v,
    resolution: v.display_width && v.display_height ? `${v.display_width}\u00d7${v.display_height}` : null,
    codedSize: v.width && v.height ? `${v.width}\u00d7${v.height}` : null,
    fps: isNumber(meta.frame_rate_float) ? meta.frame_rate_float : null,
    duration,
    codec: v.codec ? `${v.codec.toUpperCase()}${v.profile ? ` (${v.profile})` : ''}` : null,
    pixel: v.pix_fmt ? `${v.pix_fmt}${v.bit_depth ? ` \u00b7 ${v.bit_depth}-bit` : ''}` : null,
    hdr: hdrKind || (hasHdrMeta ? 'HDR' : null),
    rotation: v.rotation ? v.rotation : 0,
    interlaced: v.field_order && !['progressive', 'unknown'].includes(v.field_order) ? v.field_order : null,
    audio: audioSummary(meta.audio || []),
    warnings: Array.isArray(meta.warnings) ? meta.warnings : [],
  };
}

function describeAudioStream(stream) {
  const parts = [String(stream.codec || 'audio').toUpperCase()];
  if (stream.sample_rate) parts.push(`${formatTrim(stream.sample_rate / 1000, 1)} kHz`);
  if (stream.channel_layout) parts.push(stream.channel_layout);
  else if (stream.channels) parts.push(`${stream.channels} ch`);
  if (stream.bit_rate) parts.push(`${Math.round(stream.bit_rate / 1000)} kb/s`);
  if (stream.language && stream.language !== 'und') parts.push(stream.language);
  return parts.join(' \u00b7 ');
}

function audioSummary(streams) {
  if (!streams.length) return 'No audio';
  if (streams.length === 1) return describeAudioStream(streams[0]);
  return `${streams.length} tracks: ${streams.map((stream) => String(stream.codec || 'audio').toUpperCase()).join(', ')}`;
}

function thumbIndex(video) {
  const meta = video.metadata || {};
  const count = meta.frame_count || 1;
  const second = isNumber(meta.frame_rate_float) ? Math.round(meta.frame_rate_float) : 0;
  return Math.max(0, Math.min(count - 1, second));
}

// ------------------------------------------------------------------ drop zone, uploads

function dropZone(scope) {
  const input = h('input', { type: 'file', multiple: true, class: 'sr-only', testid: 'file-input', tabIndex: -1, 'aria-hidden': 'true' });
  input.addEventListener('change', () => {
    if (input.files && input.files.length) queueUploads(Array.from(input.files));
    input.value = '';
  });
  const formats = h('div', { class: 'dropzone-formats faint' });
  const zone = h(
    'div',
    { class: 'dropzone', testid: 'dropzone', role: 'group', 'aria-label': 'Add videos' },
    icon('upload', 'xl'),
    h('div', { class: 'dropzone-title' }, 'Drop videos here'),
    h('div', { class: 'dropzone-sub' }, 'or'),
    button({ label: 'Choose files', icon: 'folder', variant: 'primary', onClick: () => input.click(), testid: 'choose-files' }),
    formats,
    input,
  );

  scope.add(
    store.watch(
      (state) => state.options,
      (options) => {
        if (options && Array.isArray(options.input_extensions)) {
          input.accept = options.input_extensions.join(',');
          formats.textContent = `Supported: ${options.input_extensions.map((ext) => ext.replace('.', '').toUpperCase()).join(', ')}. One file for a single job, several for a batch.`;
        } else {
          input.removeAttribute('accept');
          formats.textContent = 'One file for a single job, several for a batch.';
        }
      },
    ),
  );

  // drag and drop: the whole window accepts files while this view is open
  let depth = 0;
  const hasFiles = (event) => event.dataTransfer && Array.from(event.dataTransfer.types || []).includes('Files');
  scope.on(window, 'dragenter', (event) => {
    if (!hasFiles(event)) return;
    depth += 1;
    zone.classList.add('is-dragover');
  });
  scope.on(window, 'dragleave', (event) => {
    if (!hasFiles(event)) return;
    depth = Math.max(0, depth - 1);
    if (depth === 0) zone.classList.remove('is-dragover');
  });
  scope.on(window, 'dragover', (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
  });
  scope.on(window, 'drop', (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    depth = 0;
    zone.classList.remove('is-dragover');
    const files = Array.from(event.dataTransfer.files || []);
    if (files.length) queueUploads(files);
  });
  return zone;
}

function pathImport(scope) {
  const input = h('input', { class: 'input mono', type: 'text', placeholder: 'C:\\footage\\interview.mov', spellcheck: false, autocomplete: 'off', 'aria-label': 'Path of a video file on this computer', testid: 'import-path' });
  const errorSlot = h('div');
  const add = button({ label: 'Add', icon: 'link', testid: 'import-add' });
  const submit = () => {
    const path = input.value.trim();
    if (!path) {
      input.focus();
      return;
    }
    fill(errorSlot);
    busyWhile(add, async () => {
      try {
        await importPath(path);
        input.value = '';
      } catch (error) {
        fill(errorSlot, errorNote(error, { testid: 'import-error' }));
      }
    });
  };
  add.addEventListener('click', submit);
  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') submit();
  });
  const el = h(
    'div',
    { class: 'path-import', testid: 'path-import' },
    h('div', { class: 'section-title' }, 'Use a file that is already on this computer'),
    h('div', { class: 'input-group' }, input, add),
    h('div', { class: 'field-hint' }, 'The file is registered where it is: nothing is copied and it is never modified.'),
    errorSlot,
  );
  // hidden only when the server says path import is disabled
  scope.add(
    store.watch(
      (state) => state.options,
      (options) => {
        el.hidden = !!options && options.allow_path_import === false;
      },
    ),
  );
  return el;
}

/** One row of the upload list. It is updated in place so its buttons stay clickable during progress. */
function uploadRow(key, name) {
  const bar = progressBar({ testid: 'upload-progress', label: `Upload of ${name}` });
  const status = h('div', { class: 'upload-status num faint', testid: 'upload-status' });
  const errorSlot = h('div');
  const close = button({ icon: 'x', size: 'small', variant: 'ghost', testid: 'upload-cancel' });
  let failed = false;
  close.addEventListener('click', () => (failed ? dismissUpload(key) : cancelUpload(key)));
  const el = h(
    'li',
    { class: 'upload-item', testid: 'upload-item' },
    h('div', { class: 'row between' }, h('span', { class: 'truncate upload-name' }, name), close),
    bar.el,
    status,
    errorSlot,
  );
  let shownError = null;
  return {
    el,
    update(item) {
      const percent = item.size ? (item.loaded / item.size) * 100 : 0;
      el.dataset.status = item.status;
      failed = item.status === 'error';
      close.setAttribute('data-tip', failed ? 'Dismiss' : 'Cancel this upload');
      close.setAttribute('aria-label', failed ? 'Dismiss' : 'Cancel this upload');
      bar.setTone(failed ? 'err' : '');
      if (item.status === 'waiting') {
        status.textContent = `Waiting \u00b7 ${formatBytes(item.size)}`;
        bar.set(0);
      } else if (item.status === 'uploading') {
        status.textContent = `Uploading \u00b7 ${formatBytes(item.loaded)} of ${formatBytes(item.size)} \u00b7 ${percent.toFixed(0)}%`;
        bar.set(percent);
      } else if (item.status === 'analysing') {
        status.textContent = `Uploaded ${formatBytes(item.size)} \u00b7 the server is analysing the file\u2026`;
        bar.set(null);
      } else {
        status.textContent = 'Not added';
        bar.set(percent);
      }
      if (item.error !== shownError) {
        shownError = item.error;
        fill(errorSlot, item.error ? errorNote(item.error, { testid: 'upload-error' }) : null);
      }
    },
  };
}

function uploadsList(scope) {
  const list = h('ul', { class: 'upload-list', testid: 'upload-list' });
  const wrap = h('div', { class: 'stack tight', hidden: true }, h('div', { class: 'section-title' }, 'Uploads'), list);
  const rows = new Map();
  scope.add(
    store.watch(
      (state) => state.uploads,
      (uploads) => {
        wrap.hidden = uploads.length === 0;
        const keys = new Set(uploads.map((item) => item.key));
        for (const [key, row] of rows) {
          if (!keys.has(key)) {
            row.el.remove();
            rows.delete(key);
          }
        }
        for (const item of uploads) {
          let row = rows.get(item.key);
          if (!row) {
            row = uploadRow(item.key, item.name);
            rows.set(item.key, row);
            list.appendChild(row.el);
          }
          row.update(item);
        }
      },
    ),
  );
  return wrap;
}

// ------------------------------------------------------------------ library

function propertyCell(label, value, extra = null, testid = null) {
  return h('div', { class: 'vprop' }, h('span', { class: 'vprop-label' }, label), h('span', { class: 'vprop-value', testid }, value || '\u2013', extra));
}

async function confirmDelete(video) {
  const state = store.get();
  const jobs = jobsOfVideo(state, video.id);
  const ok = await confirmDialog({
    title: 'Remove this video?',
    message: `\u201c${video.filename}\u201d is removed from the library${jobs.length ? ` together with its ${plural(jobs.length, 'job')} and their results` : ''}.`,
    detail: video.imported ? 'The original file on your disk is not touched.' : 'The uploaded copy is deleted from the data folder.',
    confirmLabel: 'Remove',
    danger: true,
  });
  if (!ok) return;
  try {
    await deleteVideo(video.id);
  } catch (error) {
    toastError(error, 'The video could not be removed');
  }
}

function videoCard(video, state) {
  const facts = videoFacts(video);
  const selected = state.selectedIds.includes(video.id);
  const multi = state.selectedIds.length > 1;
  const active = state.activeId === video.id || (!state.activeId && state.selectedIds[0] === video.id);

  const thumb = h('div', { class: 'vthumb' });
  const img = h('img', { alt: '', loading: 'lazy', decoding: 'async', src: api.frameUrl(video.id, thumbIndex(video), 320) });
  img.addEventListener('error', () => fill(thumb, h('div', { class: 'vthumb-missing', tip: 'No thumbnail available' }, icon('film', 'l'))));
  thumb.appendChild(img);
  if (isNumber(facts.duration)) thumb.appendChild(h('span', { class: 'vthumb-duration num' }, formatDuration(facts.duration)));

  const check = h('input', { type: 'checkbox', checked: selected, 'aria-label': `Select ${video.filename} for processing`, testid: 'video-select' });
  check.addEventListener('change', () => toggleSelected(video.id));

  const badges = [];
  if (facts.hdr) badges.push(badge(facts.hdr, 'accent', null, 'High dynamic range: pixel values and colour tags are passed through unchanged'));
  if (facts.rotation) badges.push(badge(`Rotated ${facts.rotation}\u00b0`, '', null, `Stored as ${facts.codedSize}, displayed rotated by ${facts.rotation} degrees`));
  if (facts.interlaced) badges.push(badge('Interlaced', 'warn', null, `Field order ${facts.interlaced}`));
  if (facts.v.has_alpha) badges.push(badge('Alpha', '', null, 'The source has an alpha channel'));
  if (video.imported) badges.push(badge('Linked file', '', 'link', video.path));

  const el = h(
    'li',
    {
      class: ['vcard', selected && 'is-selected', selected && multi && active && 'is-active'],
      testid: 'video-card',
      dataset: { videoId: video.id, selected: String(selected) },
      'aria-selected': String(selected),
    },
    h('label', { class: 'check vcard-check', tip: 'Add to the selection (several videos are processed as a batch)' }, check),
    thumb,
    h(
      'div',
      { class: 'vcard-main' },
      h(
        'div',
        { class: 'vcard-head' },
        h('span', { class: 'vcard-name truncate', testid: 'video-name', tip: video.filename }, video.filename),
        h('div', { class: 'badge-row' }, badges),
      ),
      h(
        'div',
        { class: 'vprops' },
        propertyCell('Resolution', facts.resolution, null, 'video-resolution'),
        propertyCell('Frame rate', facts.fps !== null ? formatFps(facts.fps) : null, facts.meta.is_vfr ? badge('VFR', 'warn', null, 'Variable frame rate: every frame keeps its exact timestamp') : null, 'video-fps'),
        propertyCell('Duration', isNumber(facts.duration) ? `${formatDuration(facts.duration)} \u00b7 ${formatInt(facts.meta.frame_count)} frames` : null, null, 'video-duration'),
        propertyCell('Codec', facts.codec, null, 'video-codec'),
        propertyCell('Pixel format', facts.pixel, null, 'video-pixfmt'),
        propertyCell('Audio', facts.audio, null, 'video-audio'),
      ),
      facts.warnings.length
        ? h(
            'ul',
            { class: 'vcard-warnings', testid: 'video-warnings' },
            facts.warnings.map((text) => h('li', null, icon('alert', 's'), h('span', { class: 'break' }, text))),
          )
        : null,
    ),
    h(
      'div',
      { class: 'vcard-actions' },
      button({ icon: 'trash', variant: 'ghost', size: 'small', tip: 'Remove from the library', testid: 'video-delete', onClick: () => confirmDelete(video) }),
    ),
  );

  el.addEventListener('click', (event) => {
    if (event.target.closest('button, a, input, label')) return;
    const current = store.get();
    if (current.selectedIds.length > 1 && current.selectedIds.includes(video.id)) setActive(video.id);
    else selectOnly(video.id);
  });
  return el;
}

function library(scope) {
  const list = h('ul', { class: 'vlist', testid: 'library-list' });
  const updateList = keyedList(list);
  const count = h('span', { class: 'card-sub', testid: 'library-count' });
  const body = h('div');
  const selectAll = button({ label: 'Select all', size: 'small', variant: 'ghost', onClick: () => selectMany((store.get().videos || []).map((video) => video.id)), testid: 'select-all' });
  const clear = button({ label: 'Clear selection', size: 'small', variant: 'ghost', onClick: () => selectMany([]), testid: 'clear-selection' });

  scope.add(
    store.watch(
      (state) => [state.videos, state.videosError, state.selectedIds, state.activeId],
      () => {
        const state = store.get();
        const videos = state.videos;
        count.textContent = videos ? plural(videos.length, 'video') : '';
        selectAll.hidden = !videos || videos.length < 2 || state.selectedIds.length === videos.length;
        clear.hidden = state.selectedIds.length === 0;
        if (!videos) {
          fill(
            body,
            state.videosError
              ? errorNote(state.videosError, {
                  title: 'The library could not be loaded',
                  actions: [button({ label: 'Retry', icon: 'refresh', size: 'small', onClick: () => loadVideos() })],
                  testid: 'library-error',
                })
              : skeletonCards(2),
          );
          return;
        }
        if (videos.length === 0) {
          fill(
            body,
            emptyState({
              iconName: 'film',
              title: 'No videos yet',
              text: 'Drop a video above or choose files. Videos you add stay in the library until you remove them.',
              compact: true,
              plain: true,
              testid: 'library-empty',
            }),
          );
          return;
        }
        const multi = state.selectedIds.length > 1;
        const active = activeVideo(state);
        updateList(
          videos,
          (video) => video.id,
          (video) => JSON.stringify([video, state.selectedIds.includes(video.id), multi, multi && active === video]),
          (video) => videoCard(video, state),
        );
        if (body.firstChild !== list) fill(body, list);
      },
    ),
  );

  return card({
    title: ['Library', count],
    actions: h('div', { class: 'btn-row' }, selectAll, clear),
    body,
    flush: true,
    testid: 'library',
  });
}

// ------------------------------------------------------------------ preview of the original

function detailRows(video) {
  const facts = videoFacts(video);
  const { meta, v } = facts;
  const color = v.color || {};
  const audio = (meta.audio || []).map((stream) => `#${stream.index} ${describeAudioStream(stream)}`);
  return [
    ['File', video.filename],
    ['Size', formatBytes(video.size_bytes)],
    ['Container', meta.format_long_name || meta.container || meta.format_name],
    ['Video', v.codec_long || v.codec],
    ['Picture', facts.rotation ? `${facts.codedSize} stored, ${facts.resolution} displayed (rotation ${facts.rotation}\u00b0)` : facts.resolution],
    ['Pixel format', facts.pixel],
    ['Frames', isNumber(meta.frame_count) ? formatInt(meta.frame_count) : null],
    ['Frame rate', facts.fps !== null ? `${formatFps(facts.fps)}${meta.frame_rate ? ` (${meta.frame_rate})` : ''}${meta.is_vfr ? ', variable' : ', constant'}` : null],
    ['Time base', v.time_base],
    ['Bit rate', isNumber(meta.bit_rate) ? `${formatTrim(meta.bit_rate / 1e6, 2)} Mb/s` : null],
    ['Colour', [color.range && `range ${color.range}`, color.space && `matrix ${color.space}`, color.primaries && `primaries ${color.primaries}`, color.transfer && `transfer ${color.transfer}`].filter(Boolean).join(', ')],
    ['Field order', facts.interlaced],
    ['Audio', audio.length ? audio.join('\n') : 'none'],
    ['Other streams', meta.subtitle_streams || meta.data_streams ? `${meta.subtitle_streams || 0} subtitle, ${meta.data_streams || 0} data` : null],
    ['Location', video.imported ? video.path : null],
    ['Added', formatDateTime(video.created_at)],
  ];
}

function previewPanel(scope) {
  const title = h('h2', { class: 'truncate' }, 'Preview');
  const stageSlot = h('div', { class: 'preview-slot' });
  const infoSlot = h('div', { class: 'stack' });
  let player = null;
  let shown = '';

  const dropPlayer = () => {
    if (player) {
      player.destroy();
      player = null;
    }
  };
  scope.add(dropPlayer);

  const poster = (video, message, spinner) =>
    h(
      'div',
      { class: 'poster', testid: 'preview-poster' },
      (() => {
        const img = h('img', { alt: '', src: api.frameUrl(video.id, thumbIndex(video), 960) });
        img.addEventListener('error', () => img.remove());
        return img;
      })(),
      message ? h('div', { class: 'poster-state' }, spinner ? h('span', { class: 'spinner' }) : icon('info'), h('span', null, message)) : null,
    );

  scope.add(
    store.watch(
      (state) => {
        const video = activeVideo(state);
        return [video, state.videos === null];
      },
      ([video, loading]) => {
        if (!video) {
          dropPlayer();
          shown = '';
          title.textContent = 'Preview';
          fill(
            stageSlot,
            emptyState({
              iconName: 'eye',
              title: loading ? 'Loading' : 'Nothing selected',
              text: loading ? '' : 'Select a video in the library to preview the original and see its technical details.',
              compact: true,
              plain: true,
              testid: 'preview-empty',
            }),
          );
          fill(infoSlot);
          return;
        }
        title.textContent = video.filename;
        const facts = videoFacts(video);
        const key = `${video.id}:${video.proxy}`;
        if (key !== shown) {
          shown = key;
          dropPlayer();
          if (video.proxy === 'ready' && video.preview_url) {
            player = createPlayer({
              src: video.preview_url,
              atlas: false,
              fps: facts.fps,
              frameCount: facts.meta.frame_count || null,
              vfr: !!facts.meta.is_vfr,
              hasAudio: facts.meta.has_audio !== false,
              testid: 'source-player',
            });
            fill(stageSlot, player.el);
          } else if (video.proxy === 'pending') {
            fill(stageSlot, poster(video, 'Preparing the browser preview\u2026', true));
          } else if (video.proxy === 'failed') {
            fill(
              stageSlot,
              poster(video, null, false),
              note({
                tone: 'warn',
                title: 'The browser preview could not be created',
                text: video.proxy_error || '',
                hint: 'This only affects playback in the browser. The video can still be processed.',
                testid: 'proxy-failed',
              }),
            );
          } else {
            fill(
              stageSlot,
              poster(video, null, false),
              note({ tone: 'info', text: 'No browser preview is available for this video. It can still be processed.', testid: 'proxy-unavailable' }),
            );
          }
        }
        fill(
          infoSlot,
          facts.warnings.length ? note({ tone: 'warn', title: plural(facts.warnings.length, 'note') + ' about this file', children: bullets(facts.warnings), testid: 'preview-warnings' }) : null,
          disclosure('Technical details', kv(detailRows(video), 'video-details'), { open: true, testid: 'video-details-box' }),
        );
      },
    ),
  );

  return card({
    title,
    body: h('div', { class: 'stack' }, stageSlot, infoSlot),
    testid: 'preview-panel',
    cls: 'preview-card',
  });
}

// ------------------------------------------------------------------ view

export function sourceView(scope) {
  const next = button({ label: 'Next: subject', iconRight: 'arrow-right', variant: 'primary', onClick: () => navigate('subject'), testid: 'next-subject' });
  scope.add(
    store.watch(
      (state) => state.selectedIds.length,
      (count) => {
        next.disabled = count === 0;
      },
    ),
  );

  return h(
    'div',
    { class: 'source', testid: 'view-source' },
    h(
      'div',
      { class: 'view-head' },
      h(
        'div',
        { class: 'view-title' },
        h('h1', null, 'Source'),
        h('p', null, 'Add the videos to process. Resolution, frame timing, colour and audio of the source are analysed here and preserved in the output.'),
      ),
      h('div', { class: 'view-actions' }, next),
    ),
    h(
      'div',
      { class: 'source-layout' },
      h('div', { class: 'stack' }, card({ body: h('div', { class: 'stack' }, dropZone(scope), pathImport(scope), uploadsList(scope)), testid: 'add-panel' }), library(scope)),
      h('div', { class: 'source-side' }, previewPanel(scope)),
    ),
  );
}
