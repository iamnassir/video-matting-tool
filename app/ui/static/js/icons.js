// Inline SVG icons (24 x 24 grid, drawn with strokes in the current text colour).
// Shapes are listed as path data, ['c', cx, cy, r] circles or ['r', x, y, w, h, rx] rectangles;
// a leading '!' marks a filled shape.  Nothing is loaded from the network.

import { s } from './dom.js';

const ICONS = {
  upload: ['M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4', 'M17 8l-5-5-5 5', 'M12 3v12'],
  download: ['M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4', 'M7 10l5 5 5-5', 'M12 15V3'],
  film: [['r', 2, 2, 20, 20, 2.2], 'M7 2v20', 'M17 2v20', 'M2 12h20', 'M2 7h5', 'M2 17h5', 'M17 17h5', 'M17 7h5'],
  folder: ['M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z'],
  trash: ['M3 6h18', 'M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6', 'M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2', 'M10 11v6', 'M14 11v6'],
  check: ['M20 6L9 17l-5-5'],
  x: ['M18 6L6 18', 'M6 6l12 12'],
  alert: ['M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z', 'M12 9v4', 'M12 17h.01'],
  info: [['c', 12, 12, 10], 'M12 16v-4', 'M12 8h.01'],
  'check-circle': ['M22 11.08V12a10 10 0 1 1-5.93-9.14', 'M22 4L12 14.01l-3-3'],
  'x-circle': [['c', 12, 12, 10], 'M15 9l-6 6', 'M9 9l6 6'],
  'minus-circle': [['c', 12, 12, 10], 'M8 12h8'],
  help: [['c', 12, 12, 10], 'M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3', 'M12 17h.01'],
  play: ['!M6 3.5l14 8.5-14 8.5z'],
  pause: [['!r', 5.5, 4, 4.5, 16, 1], ['!r', 14, 4, 4.5, 16, 1]],
  'step-back': ['M18 19L8 12l10-7v14z', 'M5 19V5'],
  'step-forward': ['M6 5l10 7-10 7V5z', 'M19 5v14'],
  repeat: ['M17 1l4 4-4 4', 'M3 11V9a4 4 0 0 1 4-4h14', 'M7 23l-4-4 4-4', 'M21 13v2a4 4 0 0 1-4 4H3'],
  volume: ['M11 5L6 9H2v6h4l5 4V5z', 'M15.54 8.46a5 5 0 0 1 0 7.07', 'M19.07 4.93a10 10 0 0 1 0 14.14'],
  'volume-x': ['M11 5L6 9H2v6h4l5 4V5z', 'M23 9l-6 6', 'M17 9l6 6'],
  maximize: ['M8 3H5a2 2 0 0 0-2 2v3', 'M21 8V5a2 2 0 0 0-2-2h-3', 'M16 21h3a2 2 0 0 0 2-2v-3', 'M3 16v3a2 2 0 0 0 2 2h3'],
  minimize: ['M8 3v3a2 2 0 0 1-2 2H3', 'M21 8h-3a2 2 0 0 1-2-2V3', 'M3 16h3a2 2 0 0 1 2 2v3', 'M16 21v-3a2 2 0 0 1 2-2h3'],
  refresh: ['M23 4v6h-6', 'M1 20v-6h6', 'M3.51 9a9 9 0 0 1 14.85-3.36L23 10', 'M1 14l4.64 4.36A9 9 0 0 0 20.49 15'],
  undo: ['M1 4v6h6', 'M3.51 15a9 9 0 1 0 2.13-9.36L1 10'],
  sliders: ['M4 21v-7', 'M4 10V3', 'M12 21v-9', 'M12 8V3', 'M20 21v-5', 'M20 12V3', 'M1 14h6', 'M9 8h6', 'M17 16h6'],
  cpu: [['r', 4, 4, 16, 16, 2], ['r', 9, 9, 6, 6, 0.5], 'M9 1v3', 'M15 1v3', 'M9 20v3', 'M15 20v3', 'M20 9h3', 'M20 14h3', 'M1 9h3', 'M1 14h3'],
  memory: [['r', 2, 7, 20, 10, 1.5], 'M6 11v2', 'M10 11v2', 'M14 11v2', 'M18 11v2', 'M5 17v3', 'M9.5 17v3', 'M14.5 17v3', 'M19 17v3'],
  layers: ['M12 2L2 7l10 5 10-5-10-5z', 'M2 17l10 5 10-5', 'M2 12l10 5 10-5'],
  target: [['c', 12, 12, 10], ['c', 12, 12, 6], ['c', 12, 12, 2]],
  wand: ['M3 21l9.5-9.5', 'M15 4V2', 'M15 16v-2', 'M8 9h2', 'M20 9h2', 'M17.8 11.8L19 13', 'M17.8 6.2L19 5', 'M12.2 6.2L11 5'],
  pointer: ['M3 3l7.07 16.97 2.51-7.39 7.39-2.51L3 3z', 'M13 13l6 6'],
  box: ['M4 9V6a2 2 0 0 1 2-2h3', 'M15 4h3a2 2 0 0 1 2 2v3', 'M20 15v3a2 2 0 0 1-2 2h-3', 'M9 20H6a2 2 0 0 1-2-2v-3'],
  type: ['M4 7V4h16v3', 'M9 20h6', 'M12 4v16'],
  image: [['r', 3, 3, 18, 18, 2], ['c', 8.5, 8.5, 1.5], 'M21 15l-5-5L5 21'],
  plus: ['M12 5v14', 'M5 12h14'],
  minus: ['M5 12h14'],
  'chevron-right': ['M9 18l6-6-6-6'],
  'chevron-left': ['M15 18l-6-6 6-6'],
  'chevron-down': ['M6 9l6 6 6-6'],
  'arrow-right': ['M5 12h14', 'M12 5l7 7-7 7'],
  external: ['M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6', 'M15 3h6v6', 'M10 14L21 3'],
  clock: [['c', 12, 12, 10], 'M12 6v6l4 2'],
  list: ['M8 6h13', 'M8 12h13', 'M8 18h13', 'M3 6h.01', 'M3 12h.01', 'M3 18h.01'],
  grid: [['r', 3, 3, 7, 7, 1], ['r', 14, 3, 7, 7, 1], ['r', 14, 14, 7, 7, 1], ['r', 3, 14, 7, 7, 1]],
  columns: [['r', 3, 3, 18, 18, 2], 'M12 3v18'],
  wipe: ['M12 2v20', 'M8 8l-4 4 4 4', 'M16 8l4 4-4 4'],
  eye: ['M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z', ['c', 12, 12, 3]],
  zap: ['M13 2L3 14h9l-1 8 10-12h-9l1-8z'],
  package: ['M16.5 9.4l-9-5.19', 'M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z', 'M3.27 6.96L12 12.01l8.73-5.05', 'M12 22.08V12'],
  shield: ['M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z'],
  report: ['M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z', 'M14 2v6h6', 'M16 13H8', 'M16 17H8', 'M10 9H8'],
  terminal: ['M4 17l6-6-6-6', 'M12 19h8'],
  copy: [['r', 9, 9, 13, 13, 2], 'M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1'],
  sun: [['c', 12, 12, 5], 'M12 1v2', 'M12 21v2', 'M4.22 4.22l1.42 1.42', 'M18.36 18.36l1.42 1.42', 'M1 12h2', 'M21 12h2', 'M4.22 19.78l1.42-1.42', 'M18.36 5.64l1.42-1.42'],
  moon: ['M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z'],
  stop: [['r', 5, 5, 14, 14, 2]],
  link: ['M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71', 'M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71'],
  activity: ['M22 12h-4l-3 9L9 3l-3 9H2'],
  monitor: [['r', 2, 3, 20, 14, 2], 'M8 21h8', 'M12 17v4'],
  user: ['M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2', ['c', 12, 7, 4]],
  rewind: ['M11 19L2 12l9-7v14z', 'M22 19l-9-7 9-7v14z'],
  'fast-forward': ['M13 19l9-7-9-7v14z', 'M2 19l9-7-9-7v14z'],
  more: [['!c', 5, 12, 1.4], ['!c', 12, 12, 1.4], ['!c', 19, 12, 1.4]],
  search: [['c', 11, 11, 8], 'M21 21l-4.35-4.35'],
  music: ['M9 18V5l12-2v13', ['c', 6, 18, 3], ['c', 18, 16, 3]],
  render: ['M21 12a9 9 0 1 1-3.2-6.9', 'M21 3v6h-6'],
};

function shape(spec) {
  if (typeof spec === 'string') {
    const solid = spec.startsWith('!');
    return s('path', { d: solid ? spec.slice(1) : spec, class: solid ? 'solid' : null });
  }
  const [kind, ...n] = spec;
  const solid = kind.startsWith('!');
  const type = solid ? kind.slice(1) : kind;
  if (type === 'c') return s('circle', { cx: n[0], cy: n[1], r: n[2], class: solid ? 'solid' : null });
  return s('rect', { x: n[0], y: n[1], width: n[2], height: n[3], rx: n[4] || 0, class: solid ? 'solid' : null });
}

/** Build an icon. `size` adds a size class: 's' (14 px), 'l' (20 px) or 'xl' (28 px). */
export function icon(name, size = '') {
  const shapes = ICONS[name] || ICONS.info;
  return s(
    'svg',
    { class: size ? `icon ${size}` : 'icon', viewBox: '0 0 24 24', 'aria-hidden': 'true', focusable: 'false' },
    shapes.map(shape),
  );
}

export function hasIcon(name) {
  return Object.prototype.hasOwnProperty.call(ICONS, name);
}
