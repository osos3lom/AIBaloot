/**
 * Hakim — icon system.
 *
 * The interface is icon-first, so icons are part of the product rather than
 * decoration, and they ship inline instead of from a CDN. Three reasons that
 * matters here: the app is an offline-first PWA (a CDN icon font would be the
 * one thing still needing the network), the service worker already precaches
 * every script, and emoji render differently on every platform — which an
 * icon-first UI cannot afford.
 *
 * Geometry follows Lucide: a 24x24 box, ~2px stroke, round caps and joins, so
 * every glyph carries the same optical weight. `strokeWidth` is overridable for
 * the few places that need a lighter or heavier line.
 */

var HakimIcons = (function () {
  'use strict';

  /** Inner markup of each glyph, drawn in Lucide's 24x24 coordinate box. */
  var PATHS = {
    house:
      '<path d="M15 21v-8a1 1 0 0 0-1-1h-4a1 1 0 0 0-1 1v8"/>' +
      '<path d="M3 10a2 2 0 0 1 .709-1.528l7-5.999a2 2 0 0 1 2.582 0l7 5.999A2 2 0 0 1 21 10v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',

    layers:
      '<path d="M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83z"/>' +
      '<path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/>' +
      '<path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',

    calculator:
      '<rect width="16" height="20" x="4" y="2" rx="2"/>' +
      '<line x1="8" x2="16" y1="6" y2="6"/>' +
      '<line x1="16" x2="16" y1="14" y2="18"/>' +
      '<path d="M16 10h.01"/><path d="M12 10h.01"/><path d="M8 10h.01"/>' +
      '<path d="M12 14h.01"/><path d="M8 14h.01"/>' +
      '<path d="M12 18h.01"/><path d="M8 18h.01"/>',

    history:
      '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>' +
      '<path d="M3 3v5h5"/><path d="M12 7v5l4 2"/>',

    settings:
      '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/>' +
      '<circle cx="12" cy="12" r="3"/>',

    scan:
      '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/>' +
      '<path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/>' +
      '<path d="M7 12h10"/>',

    sparkles:
      '<path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/>' +
      '<path d="M20 3v4"/><path d="M22 5h-4"/><path d="M4 17v2"/><path d="M5 18H3"/>',

    camera:
      '<path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/>' +
      '<circle cx="12" cy="13" r="3"/>',

    image:
      '<rect width="18" height="18" x="3" y="3" rx="2" ry="2"/>' +
      '<circle cx="9" cy="9" r="2"/>' +
      '<path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>',

    undo:
      '<path d="M9 14 4 9l5-5"/>' +
      '<path d="M4 9h10.5a5.5 5.5 0 0 1 5.5 5.5 5.5 5.5 0 0 1-5.5 5.5H11"/>',

    retake:
      '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>',

    plus: '<path d="M5 12h14"/><path d="M12 5v14"/>',
    minus: '<path d="M5 12h14"/>',
    check: '<path d="M20 6 9 17l-5-5"/>',
    x: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',

    pencil:
      '<path d="M21.174 6.812a1 1 0 0 0-3.986-3.987L3.842 16.174a2 2 0 0 0-.5.83l-1.321 4.352a.5.5 0 0 0 .623.622l4.353-1.32a2 2 0 0 0 .83-.497z"/>' +
      '<path d="m15 5 4 4"/>',

    stats:
      '<path d="M12 16v5"/><path d="M16 14v7"/><path d="M20 10v11"/>' +
      '<path d="m22 3-8.646 8.646a.5.5 0 0 1-.708 0L9.354 8.354a.5.5 0 0 0-.707 0L2 15"/>' +
      '<path d="M4 18v3"/><path d="M8 14v7"/>',

    users:
      '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>' +
      '<path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',

    trophy:
      '<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/>' +
      '<path d="M4 22h16"/>' +
      '<path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/>' +
      '<path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/>' +
      '<path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"/>',

    sun:
      '<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/>' +
      '<path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/>' +
      '<path d="M2 12h2"/><path d="M20 12h2"/>' +
      '<path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>',

    crown:
      '<path d="M11.562 3.266a.5.5 0 0 1 .876 0L15.39 8.87a1 1 0 0 0 1.516.294L21.183 5.5a.5.5 0 0 1 .798.519l-2.834 10.246a1 1 0 0 1-.956.734H5.81a1 1 0 0 1-.957-.734L2.02 6.02a.5.5 0 0 1 .798-.519l4.276 3.664a1 1 0 0 0 1.516-.294z"/>' +
      '<path d="M5 21h14"/>',

    trash:
      '<path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/>' +
      '<path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>' +
      '<line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/>',

    ellipsis: '<circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/><circle cx="5" cy="12" r="1"/>',

    languages:
      '<path d="m5 8 6 6"/><path d="m4 14 6-6 2-3"/><path d="M2 5h12"/><path d="M7 2h1"/>' +
      '<path d="m22 22-5-10-5 10"/><path d="M14 18h6"/>',

    beaker:
      '<path d="M4.5 3h15"/><path d="M6 3v16a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V3"/><path d="M6 14h12"/>',

    info: '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',

    backspace:
      '<path d="M20 5H9l-7 7 7 7h11a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2z"/>' +
      '<path d="m12 9 6 6"/><path d="m18 9-6 6"/>',

    chevronDown: '<path d="m6 9 6 6 6-6"/>',

    newGame:
      '<path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/>'
  };

  /**
   * Markup for one icon.
   *
   * Always `aria-hidden`: an icon-only control carries its accessible name on
   * the button itself, and a second announcement of the same thing is noise.
   *
   * @param {string} name key in PATHS
   * @param {{size?: number, strokeWidth?: number, className?: string}} [options]
   */
  function svg(name, options) {
    var opts = options || {};
    var body = PATHS[name];
    if (!body) return '';
    var size = opts.size || 24;
    var stroke = opts.strokeWidth || 1.9;
    var extra = opts.className ? ' ' + opts.className : '';
    return '<svg class="icon' + extra + '" viewBox="0 0 24 24" width="' + size + '" height="' + size +
      '" fill="none" stroke="currentColor" stroke-width="' + stroke +
      '" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">' +
      body + '</svg>';
  }

  function has(name) {
    return Object.prototype.hasOwnProperty.call(PATHS, name);
  }

  /** Fill every `[data-icon]` element under `root` with its named glyph. */
  function paint(root) {
    var scope = root || document;
    scope.querySelectorAll('[data-icon]').forEach(function (host) {
      var size = parseFloat(host.getAttribute('data-icon-size'));
      var stroke = parseFloat(host.getAttribute('data-icon-stroke'));
      host.innerHTML = svg(host.getAttribute('data-icon'), {
        size: isNaN(size) ? undefined : size,
        strokeWidth: isNaN(stroke) ? undefined : stroke
      });
    });
  }

  return { PATHS: PATHS, svg: svg, has: has, paint: paint };
})();

if (typeof module !== 'undefined' && module.exports) {
  module.exports = HakimIcons;
}
