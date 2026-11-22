// Video player that paints a single hidden <video> element onto a canvas.
//
// For job results the video is a 2 x 2 atlas (top-left original, top-right subject overlay,
// bottom-left matte, bottom-right composite; every tile is exactly half the width and height).
// Because all views are cut from the same decoded frame they are frame-synchronous by
// construction. For source videos (`atlas: false`) the whole picture is shown.

import { h, fill, clamp, createScope, isEditableTarget, setPercent } from '../dom.js';
import { icon } from '../icons.js';
import { formatTimecode, isNumber } from '../format.js';
import { persist } from '../persist.js';
import { segmented } from './controls.js';

export const ATLAS_MODES = [
  { value: 'original', label: 'Original', tip: 'The source video' },
  { value: 'subject', label: 'Subject', tip: 'Segmented: the subject highlighted' },
  { value: 'matte', label: 'Matte', tip: 'The alpha matte (white = subject)' },
  { value: 'composite', label: 'Composite', tip: 'The result on the chosen background' },
  { value: 'side', label: 'Side by side', icon: 'columns', tip: 'Original | composite' },
  { value: 'wipe', label: 'Wipe', icon: 'wipe', tip: 'Before / after with a draggable divider' },
  { value: 'grid', label: 'Grid', icon: 'grid', tip: 'All four views' },
];

// tile origin (in tile units) of each quadrant of the atlas
const TILE = { original: [0, 0], subject: [1, 0], matte: [0, 1], composite: [1, 1] };
const TILE_LABEL = { original: 'Original', subject: 'Subject', matte: 'Matte', composite: 'Composite' };
const SPEEDS = [0.25, 0.5, 1, 1.5, 2];
const HAS_RVFC = typeof HTMLVideoElement !== 'undefined' && 'requestVideoFrameCallback' in HTMLVideoElement.prototype;

/**
 * Options:
 *   src         URL of the video
 *   atlas       true for the 2 x 2 job preview
 *   fps         frame rate of the video (for frame stepping and the frame counter); null if unknown
 *   frameCount  number of frames (shown next to the frame counter); null if unknown
 *   vfr         true when the frame rate varies (frame numbers are then approximate)
 *   hasAudio    false disables the volume controls
 *   modeKey     persist key that remembers the view mode
 */
export function createPlayer({ src, atlas = false, fps = null, frameCount = null, vfr = false, hasAudio = true, modeKey = 'playerMode', testid = 'player' }) {
  const scope = createScope();
  const frameRate = isNumber(fps) && fps > 0 ? fps : null;
  let mode = atlas ? persist.get(modeKey, 'composite') : 'full';
  if (atlas && !ATLAS_MODES.some((item) => item.value === mode)) mode = 'composite';
  let split = 0.5; // wipe position 0..1
  let mediaTime = null; // presentation time of the frame on screen (from requestVideoFrameCallback)
  let scrubbing = false;
  let destroyed = false;
  let rafId = 0;

  // ---------------------------------------------------------------- elements
  const video = h('video', { class: 'player-video', preload: 'auto', playsInline: true, testid: `${testid}-video` });
  video.setAttribute('aria-hidden', 'true');
  video.tabIndex = -1;
  const canvas = h('canvas', { class: 'player-canvas', testid: `${testid}-canvas` });
  const ctx = canvas.getContext('2d', { alpha: false });
  const labels = h('div', { class: 'player-labels', 'aria-hidden': 'true' });
  const wipeHandle = h(
    'div',
    { class: 'player-wipe', role: 'slider', tabIndex: 0, hidden: true, 'aria-label': 'Before / after divider', 'aria-valuemin': '0', 'aria-valuemax': '100', testid: `${testid}-wipe` },
    h('span', { class: 'player-wipe-grip' }, icon('wipe', 's')),
  );
  const overlay = h('div', { class: 'player-overlay' });
  const stage = h('div', { class: 'player-stage', testid: `${testid}-stage` }, canvas, labels, wipeHandle, overlay, video);

  const playButton = h('button', { class: 'btn ghost icon-only', type: 'button', testid: `${testid}-play` });
  const stepBack = h('button', { class: 'btn ghost icon-only', type: 'button', tip: 'Previous frame (,)', 'aria-label': 'Previous frame', testid: `${testid}-step-back` }, icon('step-back'));
  const stepForward = h('button', { class: 'btn ghost icon-only', type: 'button', tip: 'Next frame (.)', 'aria-label': 'Next frame', testid: `${testid}-step-forward` }, icon('step-forward'));
  const timeText = h('span', { class: 'player-time num', testid: `${testid}-time` }, '00:00.000');
  const durationText = h('span', { class: 'player-duration num' }, '00:00.000');
  const frameText = h('span', { class: 'player-frame num', testid: `${testid}-frame`, hidden: !frameRate });
  const seek = h('input', { class: 'range player-seek', type: 'range', min: '0', max: '0', step: 'any', value: '0', 'aria-label': 'Position', testid: `${testid}-seek` });
  const loopButton = h('button', { class: 'btn ghost icon-only', type: 'button', 'aria-pressed': 'false', tip: 'Loop (L)', 'aria-label': 'Loop', testid: `${testid}-loop` }, icon('repeat'));
  const speed = h(
    'select',
    { class: 'select player-speed', 'aria-label': 'Playback speed', tip: 'Playback speed', testid: `${testid}-speed` },
    SPEEDS.map((value) => h('option', { value: String(value), selected: value === 1 }, `${value}×`)),
  );
  const muteButton = h('button', { class: 'btn ghost icon-only', type: 'button', testid: `${testid}-mute` });
  const volume = h('input', { class: 'range player-volume', type: 'range', min: '0', max: '1', step: '0.05', value: '1', 'aria-label': 'Volume', testid: `${testid}-volume` });
  const fullscreenButton = h('button', { class: 'btn ghost icon-only', type: 'button', testid: `${testid}-fullscreen` });

  const modeControl = atlas
    ? segmented({
        options: ATLAS_MODES,
        value: mode,
        label: 'View',
        testid: `${testid}-mode`,
        onChange: (next) => setMode(next),
      })
    : null;

  const el = h(
    'div',
    { class: 'player', tabIndex: 0, testid, 'aria-label': 'Video player', dataset: { mode } },
    modeControl ? h('div', { class: 'player-modes' }, modeControl.el) : null,
    stage,
    h(
      'div',
      { class: 'player-controls' },
      seek,
      h(
        'div',
        { class: 'player-bar' },
        h('div', { class: 'player-group' }, playButton, stepBack, stepForward),
        h('div', { class: 'player-readout' }, timeText, h('span', { class: 'faint' }, '/'), durationText, frameText),
        h('div', { class: 'grow' }),
        h('div', { class: 'player-group' }, loopButton, speed, muteButton, volume, fullscreenButton),
      ),
    ),
  );

  // ---------------------------------------------------------------- drawing
  function tileSize() {
    return atlas ? [video.videoWidth / 2, video.videoHeight / 2] : [video.videoWidth, video.videoHeight];
  }

  function layout() {
    const [tw, th] = tileSize();
    if (!tw || !th) return;
    let width = tw;
    let height = th;
    if (mode === 'side') width = tw * 2;
    if (mode === 'grid') {
      width = tw * 2;
      height = th * 2;
    }
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    stage.style.setProperty('--ar', String(width / height));
    stage.style.setProperty('--split', String(split));
    // tile labels
    const names = mode === 'side' ? ['original', 'composite'] : mode === 'grid' ? ['original', 'subject', 'matte', 'composite'] : mode === 'wipe' ? ['original', 'composite'] : [];
    labels.className = `player-labels is-${mode}`;
    fill(
      labels,
      names.map((name) => h('span', { class: `player-label is-${name}` }, TILE_LABEL[name])),
    );
    wipeHandle.hidden = mode !== 'wipe';
    wipeHandle.setAttribute('aria-valuenow', String(Math.round(split * 100)));
  }

  function draw() {
    const [tw, th] = tileSize();
    if (!tw || !th || video.readyState < 2) return;
    if (!atlas) {
      ctx.drawImage(video, 0, 0, tw, th);
      return;
    }
    const tile = (name, dx, dy) => {
      const [tx, ty] = TILE[name];
      ctx.drawImage(video, tx * tw, ty * th, tw, th, dx, dy, tw, th);
    };
    if (mode === 'grid') {
      ctx.drawImage(video, 0, 0, tw * 2, th * 2);
    } else if (mode === 'side') {
      tile('original', 0, 0);
      tile('composite', tw, 0);
    } else if (mode === 'wipe') {
      const cut = Math.round(tw * split);
      if (cut > 0) ctx.drawImage(video, 0, 0, cut, th, 0, 0, cut, th);
      if (cut < tw) ctx.drawImage(video, tw + cut, th, tw - cut, th, cut, 0, tw - cut, th);
    } else {
      tile(mode, 0, 0);
    }
  }

  function currentTime() {
    return mediaTime !== null && !video.seeking ? mediaTime : video.currentTime;
  }

  function currentFrame() {
    if (!frameRate) return null;
    const t = mediaTime !== null && !video.seeking ? mediaTime : video.currentTime;
    const index = mediaTime !== null && !video.seeking ? Math.round(t * frameRate) : Math.floor(t * frameRate + 0.001);
    return frameCount ? clamp(index, 0, frameCount - 1) : Math.max(0, index);
  }

  function updateReadout() {
    const t = currentTime();
    timeText.textContent = formatTimecode(t);
    if (frameRate) {
      const frame = currentFrame();
      const last = frameCount ? ` / ${frameCount - 1}` : '';
      frameText.textContent = `${vfr ? 'Frame ≈ ' : 'Frame '}${frame}${last}`;
    }
    if (!scrubbing) {
      seek.value = String(video.currentTime);
      setPercent(seek, video.duration ? (video.currentTime / video.duration) * 100 : 0);
    }
  }

  function paint() {
    draw();
    updateReadout();
  }

  // requestVideoFrameCallback fires for every frame that is presented (also after a seek while
  // paused); where it does not exist a requestAnimationFrame loop repaints while playing.
  function onVideoFrame(now, metadata) {
    if (destroyed) return;
    mediaTime = metadata && isNumber(metadata.mediaTime) ? metadata.mediaTime : null;
    paint();
    video.requestVideoFrameCallback(onVideoFrame);
  }

  function rafLoop() {
    if (destroyed) return;
    paint();
    if (!video.paused && !video.ended) rafId = requestAnimationFrame(rafLoop);
  }

  if (HAS_RVFC) video.requestVideoFrameCallback(onVideoFrame);

  // ---------------------------------------------------------------- transport
  function togglePlay() {
    if (video.paused || video.ended) {
      if (video.ended) video.currentTime = 0;
      const promise = video.play();
      if (promise && promise.catch) promise.catch(() => {});
    } else {
      video.pause();
    }
  }

  function seekTo(time) {
    if (!isNumber(video.duration)) return;
    mediaTime = null;
    video.currentTime = clamp(time, 0, Math.max(0, video.duration - 0.0005));
  }

  function stepFrames(count) {
    if (!video.paused) video.pause();
    if (frameRate) {
      const frame = currentFrame() + count;
      // aim at the middle of the frame so rounding can never land on a neighbour
      seekTo((clamp(frame, 0, frameCount ? frameCount - 1 : Number.MAX_SAFE_INTEGER) + 0.5) / frameRate);
    } else {
      seekTo(video.currentTime + count * 0.04);
    }
  }

  function setMode(next) {
    if (!atlas || next === mode) return;
    mode = next;
    el.dataset.mode = mode;
    persist.set(modeKey, mode);
    if (modeControl) modeControl.set(mode);
    layout();
    draw();
  }

  function setSplit(value) {
    split = clamp(value, 0, 1);
    stage.style.setProperty('--split', String(split));
    wipeHandle.setAttribute('aria-valuenow', String(Math.round(split * 100)));
    draw();
  }

  function syncPlayButton() {
    const paused = video.paused || video.ended;
    fill(playButton, icon(paused ? 'play' : 'pause'));
    const label = paused ? 'Play (Space)' : 'Pause (Space)';
    playButton.setAttribute('aria-label', label);
    playButton.setAttribute('data-tip', label);
    el.dataset.state = paused ? 'paused' : 'playing';
  }

  function syncVolume() {
    const silent = video.muted || video.volume === 0;
    fill(muteButton, icon(silent ? 'volume-x' : 'volume'));
    const label = !hasAudio ? 'This video has no audio track' : silent ? 'Unmute (M)' : 'Mute (M)';
    muteButton.setAttribute('aria-label', label);
    muteButton.setAttribute('data-tip', label);
    volume.value = String(video.muted ? 0 : video.volume);
    setPercent(volume, (video.muted ? 0 : video.volume) * 100);
  }

  function syncFullscreen() {
    const active = document.fullscreenElement === el;
    fill(fullscreenButton, icon(active ? 'minimize' : 'maximize'));
    const label = active ? 'Leave full screen (F)' : 'Full screen (F)';
    fullscreenButton.setAttribute('aria-label', label);
    fullscreenButton.setAttribute('data-tip', label);
    el.classList.toggle('is-fullscreen', active);
  }

  function toggleFullscreen() {
    if (document.fullscreenElement === el) document.exitFullscreen().catch(() => {});
    else if (el.requestFullscreen) el.requestFullscreen().catch(() => {});
  }

  function showOverlay(kind, text = '') {
    if (kind === 'none') {
      fill(overlay);
      overlay.dataset.kind = 'none';
      return;
    }
    overlay.dataset.kind = kind;
    if (kind === 'loading') fill(overlay, h('span', { class: 'spinner large' }));
    else fill(overlay, h('div', { class: 'player-message', role: 'alert', testid: `${testid}-error` }, icon('alert', 'l'), h('span', null, text)));
  }

  // ---------------------------------------------------------------- events
  playButton.addEventListener('click', togglePlay);
  stepBack.addEventListener('click', () => stepFrames(-1));
  stepForward.addEventListener('click', () => stepFrames(1));
  loopButton.addEventListener('click', () => {
    video.loop = !video.loop;
    loopButton.setAttribute('aria-pressed', String(video.loop));
    loopButton.classList.toggle('is-on', video.loop);
  });
  speed.addEventListener('change', () => {
    video.playbackRate = Number(speed.value) || 1;
  });
  muteButton.addEventListener('click', () => {
    video.muted = !video.muted;
    if (!video.muted && video.volume === 0) video.volume = 0.5;
  });
  volume.addEventListener('input', () => {
    video.volume = Number(volume.value);
    video.muted = video.volume === 0;
  });
  fullscreenButton.addEventListener('click', toggleFullscreen);

  seek.addEventListener('pointerdown', () => {
    scrubbing = true;
  });
  seek.addEventListener('input', () => {
    scrubbing = true;
    setPercent(seek, video.duration ? (Number(seek.value) / video.duration) * 100 : 0);
    seekTo(Number(seek.value));
  });
  const endScrub = () => {
    scrubbing = false;
  };
  seek.addEventListener('change', endScrub);
  seek.addEventListener('pointerup', endScrub);
  seek.addEventListener('blur', endScrub);

  video.addEventListener('loadedmetadata', () => {
    seek.max = String(video.duration || 0);
    durationText.textContent = formatTimecode(video.duration);
    layout();
  });
  video.addEventListener('loadeddata', () => {
    showOverlay('none');
    layout();
    paint();
  });
  video.addEventListener('seeked', () => {
    if (!HAS_RVFC) paint();
    else updateReadout();
  });
  video.addEventListener('seeking', updateReadout);
  video.addEventListener('timeupdate', () => {
    if (!HAS_RVFC) paint();
  });
  video.addEventListener('play', () => {
    syncPlayButton();
    if (!HAS_RVFC) {
      cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(rafLoop);
    }
  });
  video.addEventListener('pause', syncPlayButton);
  video.addEventListener('ended', syncPlayButton);
  video.addEventListener('volumechange', syncVolume);
  video.addEventListener('waiting', () => {
    if (!video.paused) showOverlay('loading');
  });
  video.addEventListener('playing', () => showOverlay('none'));
  video.addEventListener('canplay', () => {
    if (overlay.dataset.kind === 'loading') showOverlay('none');
  });
  video.addEventListener('error', () => {
    const code = video.error ? video.error.code : 0;
    const reason = { 1: 'loading was aborted', 2: 'a network error occurred', 3: 'the video could not be decoded', 4: 'the browser cannot play this file' }[code] || 'unknown error';
    showOverlay('error', `The preview video could not be loaded (${reason}).`);
  });

  // the stage: click toggles playback; in wipe mode dragging moves the divider instead
  let dragging = false;
  const splitFromEvent = (event) => {
    const rect = canvas.getBoundingClientRect();
    return rect.width ? (event.clientX - rect.left) / rect.width : 0.5;
  };
  stage.addEventListener('pointerdown', (event) => {
    if (event.button !== 0 || mode !== 'wipe') return;
    dragging = true;
    stage.setPointerCapture(event.pointerId);
    setSplit(splitFromEvent(event));
    event.preventDefault();
  });
  stage.addEventListener('pointermove', (event) => {
    if (dragging) setSplit(splitFromEvent(event));
  });
  const endDrag = (event) => {
    if (!dragging) return;
    dragging = false;
    if (stage.hasPointerCapture(event.pointerId)) stage.releasePointerCapture(event.pointerId);
  };
  stage.addEventListener('pointerup', endDrag);
  stage.addEventListener('pointercancel', endDrag);
  stage.addEventListener('click', (event) => {
    if (mode === 'wipe' || event.target.closest('.player-wipe')) return;
    togglePlay();
  });
  stage.addEventListener('dblclick', (event) => {
    if (mode !== 'wipe') {
      event.preventDefault();
      toggleFullscreen();
    }
  });
  wipeHandle.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      setSplit(split + (event.key === 'ArrowLeft' ? -0.02 : 0.02));
      event.preventDefault();
      event.stopPropagation();
    }
  });

  // keyboard shortcuts: active while the pointer or the focus is on the player, or when the page
  // focus is not on another control
  function onKey(event) {
    if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey) return;
    if (!el.isConnected || document.querySelector('dialog[open]')) return;
    const target = event.target instanceof Element ? event.target : document.body;
    if (isEditableTarget(target)) return;
    const inside = el.contains(target);
    const onPage = target === document.body || target === document.documentElement || target.id === 'view';
    if (!inside && !onPage) return;
    const interactive = target.closest('button, a, input, select, summary, [role="slider"], [role="radio"]');
    const key = event.key;
    if (key === ' ' || key === 'k') {
      // Space on a control outside the player keeps its normal meaning
      if (interactive && !inside) return;
      togglePlay();
    } else if (key === ',' || key === '<') stepFrames(-1);
    else if (key === '.' || key === '>') stepFrames(1);
    else if (key === 'ArrowLeft' || key === 'ArrowRight') {
      if (interactive && interactive.matches('input, select, [role="slider"], [role="radio"]')) return;
      seekTo(video.currentTime + (key === 'ArrowLeft' ? -1 : 1) * (event.shiftKey ? 10 : 1));
    } else if (key === 'ArrowUp' || key === 'ArrowDown') {
      if (!inside || (interactive && interactive.matches('input, select, [role="slider"], [role="radio"]'))) return;
      video.muted = false;
      video.volume = clamp(video.volume + (key === 'ArrowUp' ? 0.05 : -0.05), 0, 1);
    } else if (key === 'Home') seekTo(0);
    else if (key === 'End') seekTo(video.duration || 0);
    else if (key === 'l') loopButton.click();
    else if (key === 'm') muteButton.click();
    else if (key === 'f') toggleFullscreen();
    else if (atlas && /^[1-7]$/.test(key)) setMode(ATLAS_MODES[Number(key) - 1].value);
    else return;
    event.preventDefault();
  }
  scope.on(document, 'keydown', onKey);
  scope.on(document, 'fullscreenchange', syncFullscreen);

  // ---------------------------------------------------------------- start
  if (!hasAudio) {
    muteButton.disabled = true;
    volume.disabled = true;
  }
  syncPlayButton();
  syncVolume();
  syncFullscreen();
  showOverlay('loading');
  video.src = src;

  return {
    el,
    video,
    setMode,
    destroy() {
      if (destroyed) return;
      destroyed = true;
      cancelAnimationFrame(rafId);
      scope.dispose();
      try {
        video.pause();
        video.removeAttribute('src');
        video.load();
      } catch {
        // ignore: the element is being discarded
      }
    },
  };
}
