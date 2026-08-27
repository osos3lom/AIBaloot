/**
 * Hakim — shared UI primitives.
 *
 * The plumbing every surface needs and none of them should own a private copy
 * of: overlay lifecycle with focus handling, toasts, haptics, the counting
 * score animation, and the two places `requestAnimationFrame` needs a guard.
 *
 * Deliberately knows nothing about Baloot. It moves focus, paints numbers, and
 * buzzes — the app and the scanner supply the meaning.
 */

var HakimUI = (function () {
  'use strict';

  function $(id) { return document.getElementById(id); }

  // ---- Device feedback -------------------------------------------------

  /** Best-effort tactile confirmation; silently absent on desktop and iOS. */
  function buzz(pattern) {
    if (navigator.vibrate) {
      try { navigator.vibrate(pattern || 8); } catch (err) { /* ignore */ }
    }
  }

  var toastTimer = null;

  function toast(message) {
    var existing = document.querySelector('.toast');
    if (existing) existing.remove();
    if (toastTimer) clearTimeout(toastTimer);

    var node = document.createElement('div');
    node.className = 'toast';
    node.setAttribute('role', 'status');
    node.textContent = message;
    document.body.appendChild(node);
    toastTimer = setTimeout(function () { node.remove(); }, 1800);
  }

  // ---- Installed-app chrome ----------------------------------------------

  /** Paper, and the scanner's ink — the app's only two full-screen grounds. */
  var THEME_PAPER = '#f2efe7';
  var THEME_INK = '#131518';

  /**
   * Repaint the system chrome to match the surface underneath it.
   *
   * Installed on a phone there is no browser frame, so the status bar and the
   * Android navigation bar are drawn in `theme-color`. A fixed ivory value
   * would leave a bright band above the scanner, which is full-bleed ink — the
   * one place in the app where the ground changes completely.
   */
  function setThemeColor(color) {
    var meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute('content', color);
  }

  function useDarkChrome() { setThemeColor(THEME_INK); }
  function usePaperChrome() { setThemeColor(THEME_PAPER); }

  // ---- Storage ----------------------------------------------------------

  /** Reads and writes that must never throw: private mode blocks both. */
  function readFlag(key) {
    try { return window.localStorage.getItem(key); } catch (err) { return null; }
  }

  function writeFlag(key, value) {
    try { window.localStorage.setItem(key, value); } catch (err) { /* ignore */ }
  }

  // ---- Frame timing -------------------------------------------------------

  /**
   * Yield once so pending paint lands before heavy work starts.
   *
   * `requestAnimationFrame` is right while the page is visible, but a hidden
   * page never fires it — and a page can be hidden at exactly this moment,
   * because returning from the native camera picker is a visibility
   * transition. Falling back to a timer means the work runs instead of hanging
   * behind an overlay that never goes away.
   */
  function afterPaint(callback) {
    if (document.hidden) setTimeout(callback, 0);
    else requestAnimationFrame(callback);
  }

  function prefersReducedMotion() {
    return Boolean(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  }

  /**
   * Count a score up or down instead of swapping the number.
   *
   * The point is legibility, not decoration: at a table the player looks up
   * mid-animation, and a number in motion says "that just changed" far more
   * clearly than a value that was already different by the time they looked.
   */
  function animateNumber(node, from, to) {
    if (prefersReducedMotion() || from === to || document.hidden) {
      node.textContent = to;
      return;
    }

    var duration = 460;
    var startedAt = null;

    function frame(now) {
      if (startedAt === null) startedAt = now;
      var progress = Math.min(1, (now - startedAt) / duration);
      var eased = 1 - Math.pow(1 - progress, 3);
      node.textContent = Math.round(from + (to - from) * eased);
      if (progress < 1) requestAnimationFrame(frame);
      else node.textContent = to;
    }

    requestAnimationFrame(frame);
  }

  /** Briefly scale a node to acknowledge a change it just received. */
  function bump(node) {
    node.setAttribute('data-bump', 'true');
    setTimeout(function () { node.removeAttribute('data-bump'); }, 420);
  }

  // ---- Overlays -------------------------------------------------------------

  var OVERLAY_SELECTOR = '.sheet:not(.hidden), .scanner:not(.hidden), .win:not(.hidden)';

  // Where focus came from, per overlay, so closing puts it back rather than
  // dropping a keyboard user at the top of the document.
  var returnFocus = {};
  var fallbackFocusId = null;

  function setFallbackFocus(id) { fallbackFocusId = id; }

  function focusables(root) {
    return Array.prototype.filter.call(
      root.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'),
      function (node) { return !node.disabled && node.offsetParent !== null; }
    );
  }

  function isOpen(id) {
    var node = $(id);
    return Boolean(node) && !node.classList.contains('hidden');
  }

  function open(id, focusTarget) {
    var node = $(id);
    if (!node || !node.classList.contains('hidden')) return;
    returnFocus[id] = document.activeElement;
    node.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
    var target = focusTarget || focusables(node)[0];
    if (target) target.focus();
  }

  function close(id) {
    var node = $(id);
    if (!node || node.classList.contains('hidden')) return;
    node.classList.add('hidden');
    if (!document.querySelector(OVERLAY_SELECTOR)) document.body.style.overflow = '';

    // The opener may have been re-rendered away while the overlay was up, which
    // would strand focus on a detached node — hence the fallback.
    var opener = returnFocus[id];
    var usable = opener && opener !== document.body && opener.isConnected &&
      typeof opener.focus === 'function';
    var target = usable ? opener : (fallbackFocusId ? $(fallbackFocusId) : null);
    if (target) target.focus();
    returnFocus[id] = null;
  }

  /** Keep Tab inside whichever overlay is on top. */
  function trapTab(event) {
    var top = document.querySelector(OVERLAY_SELECTOR);
    if (!top) return;
    var items = focusables(top);
    if (!items.length) return;

    var first = items[0];
    var last = items[items.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  return {
    $: $,
    buzz: buzz,
    toast: toast,
    readFlag: readFlag,
    writeFlag: writeFlag,
    setThemeColor: setThemeColor,
    useDarkChrome: useDarkChrome,
    usePaperChrome: usePaperChrome,
    afterPaint: afterPaint,
    prefersReducedMotion: prefersReducedMotion,
    animateNumber: animateNumber,
    bump: bump,
    isOpen: isOpen,
    open: open,
    close: close,
    trapTab: trapTab,
    focusables: focusables,
    setFallbackFocus: setFallbackFocus
  };
})();

if (typeof module !== 'undefined' && module.exports) {
  module.exports = HakimUI;
}
