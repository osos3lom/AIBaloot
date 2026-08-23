/**
 * Hakim Studio — bring-your-own-dataset lab.
 *
 * Two modes, one interface:
 *   • Served by `hakim-vision studio`, a localhost API inspects folders, builds
 *     the Baloot dataset, and runs real training jobs on this machine.
 *   • Opened as a plain file, the same screens still parse a folder the user
 *     picks in the browser and hand back the exact commands to run.
 *
 * Nothing here simulates training. When there is no backend the buttons say so.
 */
(function () {
  'use strict';

  /** Label files read before the browser-side parser stops and reports a sample. */
  var CLIENT_LABEL_LIMIT = 4000;
  /** How often to poll a running job, in milliseconds. */
  var JOB_POLL_MS = 1500;
  /** Bars drawn in the class histogram before it becomes unreadable. */
  var MAX_HISTOGRAM_BARS = 60;

  var IMAGE_PATTERN = /\.(jpe?g|png|bmp|webp)$/i;

  var STRINGS = {
    ar: {
      ds_title: 'مجموعة بياناتك: استيراد وفحص ومطابقة أصناف البلوت',
      backend_checking: 'جارٍ فحص الخادم المحلي…',
      backend_on: 'الخادم المحلي متصل — التدريب متاح',
      backend_off: 'بدون خادم — الفحص فقط، والتدريب عبر الطرفية',
      backend_unauthorised: 'الخادم يعمل بلا إذن — افتح الرابط الذي طبعه الأمر (يحوي الرمز)',
      backend_no_ultralytics: 'الخادم متصل، لكن حزمة Ultralytics غير مثبتة',
      device_line: '{device}',
      ds_path_label: 'مسار مجلد البيانات على هذا الجهاز',
      ds_path_hint: 'يقرأ الخادم المحلي المجلد مباشرة — مناسب لعشرات الآلاف من الصور.',
      ds_folder_label: 'أو اختر المجلد من المتصفح (بدون خادم)',
      ds_folder_hint: 'يُقرأ محليًا في المتصفح. المجموعات الكبيرة قد تستغرق دقيقة.',
      btn_inspect: 'افحص',
      kaggle_title: 'لا توجد بيانات؟ ابدأ بمجموعة كاجل',
      kaggle_hint: 'مجموعة أوراق اللعب (٥٢ صنفًا) — يحوّلها حكيم إلى ٣٢ صنف بلوت في الخطوة التالية.',
      kaggle_auth: 'يتطلب تسجيل الدخول في كاجل (kaggle.json). فك ضغط الملف ثم ضع مسار المجلد في الحقل المجاور.',
      btn_copy_cmd: 'انسخ الأمر',
      copied: 'تم النسخ',
      inspecting: 'جارٍ الفحص…',
      scanning_files: 'جارٍ قراءة الملفات… {done} من {total}',
      inspect_ok: 'تم فحص {images} صورة و{boxes} صندوقًا.',
      client_sampled: 'عُرضت عيّنة من {limit} ملف وسم فقط. للفحص الكامل شغّل: uv run hakim-vision studio',
      no_labels: 'لم يُعثر على ملفات وسم (.txt) في هذا المجلد.',
      stat_images: 'الصور',
      stat_boxes: 'الصناديق',
      stat_classes: 'الأصناف الموجودة',
      stat_baloot: 'تُطابق البلوت',
      splits_title: 'التقسيمات',
      th_split: 'التقسيم', th_images: 'صور', th_labelled: 'موسومة', th_boxes: 'صناديق',
      hist_title: 'توزيع الصناديق على الأصناف',
      hist_dataset_label: 'عدد الصناديق',
      map_title: 'مطابقة الأصناف مع صكة البلوت (٣٢ ورقة)',
      btn_apply_suggestions: 'طبّق الاقتراحات',
      btn_drop_all: 'أسقط الكل',
      th_source: 'صنف المصدر', th_count: 'صناديق', th_target: 'صنف البلوت', th_reason: 'السبب',
      drop_option: '— أسقطه —',
      map_summary: 'سيُبقي {kept} صنفًا و{boxes} صندوقًا، ويُسقط {dropped} صنفًا.',
      reason_exact: 'مطابق',
      reason_normalised: 'مطابق بعد التوحيد',
      reason_non_baloot: 'رتبة خارج صكة البلوت',
      reason_unknown: 'اسم غير معروف',
      reason_manual: 'اختيار يدوي',
      out_label: 'مجلد الإخراج لنسخة البلوت',
      out_hint: 'تُكتب الصور بروابط صلبة عند الإمكان، فلا تتضاعف المساحة.',
      btn_build: 'جهّز نسخة البلوت',
      building: 'جارٍ تجهيز نسخة البلوت…',
      build_done: 'جاهزة. استخدم data.yaml في خلية التدريب.',
      need_backend_build: 'تجهيز النسخة يحتاج الخادم المحلي. شغّل الأمر التالي ثم أعد المحاولة:',
      train_title: 'تدريب كاشف أوراق البلوت',
      btn_train: 'ابدأ التدريب',
      btn_training: 'التدريب يعمل…',
      btn_stop: 'أوقف',
      lbl_data_yaml: 'ملف data.yaml',
      lbl_model: 'النموذج الأساس',
      lbl_device: 'الجهاز',
      device_auto: 'تلقائي',
      lbl_epochs: 'الحقب', lbl_imgsz: 'حجم الصورة', lbl_batch: 'الدفعة',
      lbl_status: 'الحالة:', lbl_epoch: 'الحقبة:', lbl_map: 'mAP@50:',
      status_idle: 'في الانتظار',
      status_running: 'يعمل',
      status_done: 'انتهى',
      status_failed: 'فشل',
      status_stopped: 'أُوقف',
      lbl_cli: 'نفس الأمر من الطرفية',
      chart_title: 'منحنى الخسارة والدقة (من results.csv الفعلي)',
      chart_loss: 'خسارة الصندوق',
      chart_map: 'mAP@50',
      log_title: 'سجل التشغيل',
      need_backend_train: 'التدريب يحتاج الخادم المحلي. انسخ الأمر أعلاه وشغّله في الطرفية.',
      need_ultralytics: 'ثبّت حزمة التدريب أولًا: uv sync --extra train',
      export_title: 'تصدير النموذج للمتصفح (ONNX)',
      btn_export: 'صدّر ONNX',
      export_hint: 'بعد التصدير، صِل النموذج بحاسبة اليد عبر HakimDetector.registerClassifier فتتعرّف على الأوراق تلقائيًا.',
      exporting: 'جارٍ التصدير…',
      error_prefix: 'خطأ: {message}'
    },
    en: {
      ds_title: 'Your dataset: import, inspect, and map it onto the Baloot deck',
      backend_checking: 'Checking for the local server…',
      backend_on: 'Local server connected — training available',
      backend_off: 'No server — inspection only, train from the terminal',
      backend_unauthorised: 'Server running but unauthorised — open the URL the CLI printed (it carries the token)',
      backend_no_ultralytics: 'Server connected, but Ultralytics is not installed',
      device_line: '{device}',
      ds_path_label: 'Dataset folder path on this machine',
      ds_path_hint: 'The local server reads the folder directly — fine for tens of thousands of images.',
      ds_folder_label: 'Or pick the folder in the browser (no server)',
      ds_folder_hint: 'Parsed locally in your browser. Large datasets can take a minute.',
      btn_inspect: 'Inspect',
      kaggle_title: 'No data yet? Start with the Kaggle set',
      kaggle_hint: 'The 52-class playing-card dataset — Hakim maps it to the 32 Baloot classes next.',
      kaggle_auth: 'Needs a Kaggle login (kaggle.json). Unzip it, then paste the folder path on the left.',
      btn_copy_cmd: 'Copy command',
      copied: 'Copied',
      inspecting: 'Inspecting…',
      scanning_files: 'Reading files… {done} of {total}',
      inspect_ok: 'Inspected {images} images and {boxes} boxes.',
      client_sampled: 'Only {limit} label files were sampled. For a full scan run: uv run hakim-vision studio',
      no_labels: 'No label files (.txt) were found in this folder.',
      stat_images: 'Images',
      stat_boxes: 'Boxes',
      stat_classes: 'Classes present',
      stat_baloot: 'Map to Baloot',
      splits_title: 'Splits',
      th_split: 'Split', th_images: 'Images', th_labelled: 'Labelled', th_boxes: 'Boxes',
      hist_title: 'Boxes per class',
      hist_dataset_label: 'Boxes',
      map_title: 'Map source classes onto the 32-card Baloot deck',
      btn_apply_suggestions: 'Apply suggestions',
      btn_drop_all: 'Drop all',
      th_source: 'Source class', th_count: 'Boxes', th_target: 'Baloot class', th_reason: 'Why',
      drop_option: '— drop —',
      map_summary: 'Keeps {kept} classes and {boxes} boxes; drops {dropped} classes.',
      reason_exact: 'exact match',
      reason_normalised: 'matched after normalising',
      reason_non_baloot: 'rank not in the Baloot deck',
      reason_unknown: 'unrecognised name',
      reason_manual: 'set by hand',
      out_label: 'Output folder for the Baloot copy',
      out_hint: 'Images are hardlinked where possible, so disk use does not double.',
      btn_build: 'Build Baloot dataset',
      building: 'Building the Baloot dataset…',
      build_done: 'Ready. Use its data.yaml in the training cell.',
      need_backend_build: 'Building needs the local server. Run this, then try again:',
      train_title: 'Train the Baloot card detector',
      btn_train: 'Start training',
      btn_training: 'Training…',
      btn_stop: 'Stop',
      lbl_data_yaml: 'data.yaml file',
      lbl_model: 'Base model',
      lbl_device: 'Device',
      device_auto: 'Auto',
      lbl_epochs: 'Epochs', lbl_imgsz: 'Image size', lbl_batch: 'Batch',
      lbl_status: 'Status:', lbl_epoch: 'Epoch:', lbl_map: 'mAP@50:',
      status_idle: 'Idle',
      status_running: 'Running',
      status_done: 'Finished',
      status_failed: 'Failed',
      status_stopped: 'Stopped',
      lbl_cli: 'The same run from a terminal',
      chart_title: 'Loss and accuracy (from the real results.csv)',
      chart_loss: 'Box loss',
      chart_map: 'mAP@50',
      log_title: 'Run log',
      need_backend_train: 'Training needs the local server. Copy the command above and run it.',
      need_ultralytics: 'Install the training extra first: uv sync --extra train',
      export_title: 'Export the model for the browser (ONNX)',
      btn_export: 'Export ONNX',
      export_hint: 'After exporting, wire it into the hand app with HakimDetector.registerClassifier so cards are named automatically.',
      exporting: 'Exporting…',
      error_prefix: 'Error: {message}'
    }
  };

  var BALOOT_RANKS = ['A', 'K', 'Q', 'J', '10', '9', '8', '7'];
  var BALOOT_SUITS = ['h', 'd', 'c', 's'];
  var BALOOT_CLASSES = [];
  BALOOT_SUITS.forEach(function (suit) {
    BALOOT_RANKS.forEach(function (rank) { BALOOT_CLASSES.push(rank + suit); });
  });

  var state = {
    lang: document.documentElement.lang === 'en' ? 'en' : 'ar',
    backend: null,
    token: '',
    report: null,
    suggestions: [],
    mapping: {},
    source: '',
    job: null,
    unauthorised: false,
    pollTimer: null,
    chart: null,
    histogram: null
  };

  var el = {};

  function $(id) { return document.getElementById(id); }

  function t(key, vars) {
    var dict = STRINGS[state.lang] || STRINGS.ar;
    var value = dict[key] !== undefined ? dict[key] : STRINGS.ar[key];
    if (value === undefined) return key;
    if (!vars) return value;
    return value.replace(/\{(\w+)\}/g, function (match, name) {
      return Object.prototype.hasOwnProperty.call(vars, name) ? String(vars[name]) : match;
    });
  }

  function applyStrings() {
    document.querySelectorAll('[data-lab-i18n]').forEach(function (node) {
      node.textContent = t(node.getAttribute('data-lab-i18n'));
    });
  }

  // ---- API ------------------------------------------------------------

  function readToken() {
    var match = /token=([^&]+)/.exec(window.location.hash || window.location.search);
    if (match) {
      var token = decodeURIComponent(match[1]);
      try { sessionStorage.setItem('hakim_studio_token', token); } catch (error) { /* private mode */ }
      return token;
    }
    try { return sessionStorage.getItem('hakim_studio_token') || ''; } catch (error) { return ''; }
  }

  function api(path, options) {
    var settings = options || {};
    var init = { method: settings.body ? 'POST' : 'GET', headers: {} };
    if (state.token) init.headers['X-Hakim-Token'] = state.token;
    if (settings.body) {
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(settings.body);
    }
    return fetch(path, init).then(function (response) {
      return response.json().then(function (payload) {
        if (!response.ok) throw new Error(payload.error || ('HTTP ' + response.status));
        return payload;
      });
    });
  }

  function detectBackend() {
    return api('/api/status')
      .then(function (status) {
        // A server we cannot authenticate against is worse than no server: every
        // action would 403. Treat it as unusable and say why.
        state.backend = status.authenticated ? status : null;
        state.unauthorised = !status.authenticated;
        renderBackendBadge();
        populateModels(status.models || []);
        return status;
      })
      .catch(function () {
        state.backend = null;
        state.unauthorised = false;
        renderBackendBadge();
        populateModels(['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt']);
        return null;
      });
  }

  function renderBackendBadge() {
    var badge = el['backend-badge'];
    if (!badge) return;
    var text;
    var tone;
    if (state.unauthorised) {
      text = t('backend_unauthorised');
      tone = 'border-amber-500/40 bg-amber-500/10 text-amber-300';
    } else if (!state.backend) {
      text = t('backend_off');
      tone = 'border-slate-700 bg-slate-800 text-slate-300';
    } else if (!state.backend.ultralytics) {
      text = t('backend_no_ultralytics');
      tone = 'border-amber-500/40 bg-amber-500/10 text-amber-300';
    } else {
      text = t('backend_on') + ' · ' + (state.backend.device_name || 'cpu');
      tone = 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300';
    }
    badge.className = 'text-[11px] px-2.5 py-1 rounded-full border ' + tone;
    badge.textContent = text;
    updateActionAvailability();
  }

  function updateActionAvailability() {
    var hasBackend = Boolean(state.backend);
    if (el['btn-ds-remap']) el['btn-ds-remap'].disabled = !hasBackend || !state.report;
    if (el['btn-train-start']) el['btn-train-start'].disabled = !hasBackend;
    if (el['btn-export']) el['btn-export'].disabled = !hasBackend;
  }

  // ---- status line ----------------------------------------------------

  function showStatus(message, tone) {
    var box = el['ds-status'];
    if (!box) return;
    var tones = {
      info: 'border-slate-700 bg-slate-800/60 text-slate-300',
      warn: 'border-amber-500/40 bg-amber-500/10 text-amber-300',
      error: 'border-rose-500/40 bg-rose-500/10 text-rose-300',
      ok: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
    };
    box.className = 'text-xs px-3 py-2 rounded-lg border ' + (tones[tone] || tones.info);
    box.textContent = message;
    box.classList.remove('hidden');
  }

  function hideStatus() {
    if (el['ds-status']) el['ds-status'].classList.add('hidden');
  }

  // ---- client-side folder parsing -------------------------------------

  function parseNamesFromYaml(text) {
    var names = [];
    var inline = /names\s*:\s*\[(.*?)\]/s.exec(text);
    if (inline) {
      inline[1].split(',').forEach(function (part) {
        var cleaned = part.trim().replace(/^['"]|['"]$/g, '');
        if (cleaned) names.push(cleaned);
      });
      return names;
    }
    var lines = text.split(/\r?\n/);
    var collecting = false;
    var indexed = [];
    lines.forEach(function (line) {
      if (/^names\s*:/.test(line.trim())) { collecting = true; return; }
      if (!collecting) return;
      var listItem = /^\s*-\s*(.+)$/.exec(line);
      var mapItem = /^\s+(\d+)\s*:\s*(.+)$/.exec(line);
      if (listItem) {
        names.push(listItem[1].trim().replace(/^['"]|['"]$/g, ''));
      } else if (mapItem) {
        indexed.push([parseInt(mapItem[1], 10), mapItem[2].trim().replace(/^['"]|['"]$/g, '')]);
      } else if (line.trim() && !/^\s/.test(line)) {
        collecting = false;
      }
    });
    if (!names.length && indexed.length) {
      indexed.sort(function (a, b) { return a[0] - b[0]; });
      names = indexed.map(function (pair) { return pair[1]; });
    }
    return names;
  }

  function splitOf(relativePath) {
    var lower = relativePath.toLowerCase();
    if (/(^|\/)(valid|val|validation)(\/|$)/.test(lower)) return 'val';
    if (/(^|\/)test(ing)?(\/|$)/.test(lower)) return 'test';
    return 'train';
  }

  /** Build the same report shape the server returns, from a picked folder. */
  function analyseFolder(files) {
    var list = Array.prototype.slice.call(files);
    var configFile = null;
    var namesFile = null;
    var images = [];
    var labels = [];

    list.forEach(function (file) {
      var relative = (file.webkitRelativePath || file.name).replace(/\\/g, '/');
      var lower = relative.toLowerCase();
      if (/\/(data|dataset)\.ya?ml$/.test(lower) || /^(data|dataset)\.ya?ml$/.test(lower)) {
        configFile = file;
      } else if (/\/(classes|obj)\.(txt|names)$/.test(lower) || /^(classes|obj)\.(txt|names)$/.test(lower)) {
        namesFile = file;
      } else if (IMAGE_PATTERN.test(lower)) {
        images.push({ file: file, path: relative, split: splitOf(relative) });
      } else if (/\.txt$/.test(lower)) {
        labels.push({ file: file, path: relative, split: splitOf(relative) });
      }
    });

    if (!labels.length) {
      return Promise.reject(new Error(t('no_labels')));
    }

    var namesPromise = configFile
      ? configFile.text().then(parseNamesFromYaml)
      : (namesFile
        ? namesFile.text().then(function (text) {
          return text.split(/\r?\n/).map(function (line) { return line.trim(); }).filter(Boolean);
        })
        : Promise.resolve([]));

    return namesPromise.then(function (classNames) {
      var sample = labels.slice(0, CLIENT_LABEL_LIMIT);
      var splits = {};
      var counts = {};
      var totalBoxes = 0;

      images.forEach(function (image) {
        splits[image.split] = splits[image.split] || { name: image.split, images: 0, labelled_images: 0, instances: 0 };
        splits[image.split].images += 1;
      });

      var index = 0;
      function step() {
        var slice = sample.slice(index, index + 200);
        if (!slice.length) {
          return Promise.resolve();
        }
        return Promise.all(slice.map(function (entry) { return entry.file.text(); }))
          .then(function (contents) {
            contents.forEach(function (content, offset) {
              var entry = slice[offset];
              var bucket = splits[entry.split] || { name: entry.split, images: 0, labelled_images: 0, instances: 0 };
              splits[entry.split] = bucket;
              var boxes = 0;
              content.split(/\r?\n/).forEach(function (line) {
                var parts = line.trim().split(/\s+/);
                if (parts.length < 5) return;
                var classId = parseInt(parts[0], 10);
                if (isNaN(classId)) return;
                counts[classId] = (counts[classId] || 0) + 1;
                boxes += 1;
              });
              if (boxes) {
                bucket.labelled_images += 1;
                bucket.instances += boxes;
                totalBoxes += boxes;
              }
            });
            index += slice.length;
            showStatus(t('scanning_files', { done: index, total: sample.length }), 'info');
            return step();
          });
      }

      return step().then(function () {
        if (!classNames.length) {
          var highest = Object.keys(counts).reduce(function (max, key) {
            return Math.max(max, parseInt(key, 10));
          }, -1);
          classNames = [];
          for (var i = 0; i <= highest; i += 1) classNames.push('class_' + i);
        }
        return {
          report: {
            root: (list[0] && (list[0].webkitRelativePath || '').split('/')[0]) || '',
            class_names: classNames,
            splits: Object.keys(splits).map(function (key) { return splits[key]; }),
            totals: {
              images: images.length,
              instances: totalBoxes,
              classes_present: Object.keys(counts).length
            },
            class_counts: Object.keys(counts).map(function (key) {
              var id = parseInt(key, 10);
              return { id: id, name: classNames[id] || ('class_' + id), count: counts[key] };
            }).sort(function (a, b) { return a.id - b.id; }),
            issues: [],
            sample_images: []
          },
          truncated: labels.length > sample.length
        };
      });
    });
  }

  // ---- client-side class-name normalisation ---------------------------

  var RANK_WORDS = {
    a: 'A', ace: 'A', k: 'K', king: 'K', q: 'Q', queen: 'Q', j: 'J', jack: 'J',
    10: '10', t: '10', ten: '10', 9: '9', nine: '9', 8: '8', eight: '8', 7: '7', seven: '7'
  };
  var SUIT_WORDS = {
    h: 'h', heart: 'h', hearts: 'h', d: 'd', diamond: 'd', diamonds: 'd',
    c: 'c', club: 'c', clubs: 'c', s: 's', spade: 's', spades: 's'
  };

  function normaliseCardName(name) {
    var text = String(name || '').trim().toLowerCase();
    if (!text) return null;
    var compact = text.replace(/[\s_\-./]+/g, '');

    var forward = /^(10|[2-9]|[akqjt])([hdcs])$/.exec(compact);
    if (forward) {
      var rank = RANK_WORDS[forward[1]];
      var suit = SUIT_WORDS[forward[2]];
      return rank && suit ? rank + suit : null;
    }
    var reversed = /^([hdcs])(10|[2-9]|[akqjt])$/.exec(compact);
    if (reversed) {
      var suit2 = SUIT_WORDS[reversed[1]];
      var rank2 = RANK_WORDS[reversed[2]];
      return rank2 && suit2 ? rank2 + suit2 : null;
    }

    var words = text.split(/[\s_\-./]+/).filter(function (word) { return word && word !== 'of'; });
    var foundRank = null;
    var foundSuit = null;
    words.forEach(function (word) {
      if (!foundRank && RANK_WORDS[word]) { foundRank = RANK_WORDS[word]; return; }
      if (!foundSuit && SUIT_WORDS[word]) foundSuit = SUIT_WORDS[word];
    });
    return foundRank && foundSuit ? foundRank + foundSuit : null;
  }

  function suggestLocally(classNames) {
    return classNames.map(function (name, index) {
      var normalised = normaliseCardName(name);
      if (normalised && BALOOT_CLASSES.indexOf(normalised) !== -1) {
        return {
          source_id: index,
          source_name: name,
          target: normalised,
          reason: normalised === String(name).trim() ? 'exact' : 'normalised'
        };
      }
      var compact = String(name).trim().toLowerCase().replace(/[\s_\-./]+/g, '');
      var head = compact.slice(0, 2) === '10' ? '10' : compact.slice(0, 1);
      var reason = ['2', '3', '4', '5', '6'].indexOf(head) !== -1
        ? 'rank not in the 32-card Baloot deck'
        : 'unrecognised class name';
      return { source_id: index, source_name: name, target: null, reason: reason };
    });
  }

  function reasonText(reason) {
    if (reason === 'exact') return t('reason_exact');
    if (reason === 'normalised') return t('reason_normalised');
    if (reason === 'manual') return t('reason_manual');
    if (reason && reason.indexOf('Baloot deck') !== -1) return t('reason_non_baloot');
    return t('reason_unknown');
  }

  // ---- report rendering -----------------------------------------------

  function adoptReport(report, suggestions, options) {
    state.report = report;
    state.suggestions = suggestions;
    state.mapping = {};
    suggestions.forEach(function (item) { state.mapping[item.source_name] = item.target; });

    el['ds-report'].classList.remove('hidden');
    renderStats();
    renderSplits();
    renderIssues();
    renderHistogram();
    renderMapping();
    updateActionAvailability();

    var settings = options || {};
    if (settings.truncated) {
      showStatus(t('client_sampled', { limit: CLIENT_LABEL_LIMIT }), 'warn');
    } else {
      showStatus(
        t('inspect_ok', {
          images: report.totals.images,
          boxes: report.totals.instances
        }),
        'ok'
      );
    }
  }

  function renderStats() {
    var report = state.report;
    var mapped = {};
    state.suggestions.forEach(function (item) {
      if (item.target) mapped[item.target] = true;
    });
    el['stat-images'].textContent = report.totals.images;
    el['stat-boxes'].textContent = report.totals.instances;
    el['stat-classes'].textContent = report.totals.classes_present;
    el['stat-baloot'].textContent = Object.keys(mapped).length + ' / 32';
  }

  function renderSplits() {
    var body = el['splits-body'];
    body.innerHTML = '';
    state.report.splits.forEach(function (split) {
      var row = document.createElement('tr');
      row.innerHTML = '<td class="py-1.5 text-slate-200">' + split.name + '</td>' +
        '<td class="py-1.5 font-mono text-slate-300">' + split.images + '</td>' +
        '<td class="py-1.5 font-mono text-slate-300">' + split.labelled_images + '</td>' +
        '<td class="py-1.5 font-mono text-emerald-400 text-end">' + split.instances + '</td>';
      body.appendChild(row);
    });
  }

  function renderIssues() {
    var box = el['issues-list'];
    box.innerHTML = '';
    (state.report.issues || []).forEach(function (issue) {
      var node = document.createElement('p');
      node.className = 'text-[11px] text-amber-300/90 bg-amber-500/5 border border-amber-500/20 rounded-lg px-2 py-1';
      var examples = issue.examples && issue.examples.length ? ' — ' + issue.examples.slice(0, 3).join(', ') : '';
      node.textContent = issue.code + ' (' + issue.split + '): ' + issue.count + examples;
      box.appendChild(node);
    });
  }

  function renderHistogram() {
    var canvas = el['class-histogram'];
    if (!canvas || typeof Chart === 'undefined') return;
    var counts = (state.report.class_counts || []).slice()
      .sort(function (a, b) { return b.count - a.count; })
      .slice(0, MAX_HISTOGRAM_BARS);

    if (state.histogram) state.histogram.destroy();
    state.histogram = new Chart(canvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels: counts.map(function (item) { return item.name; }),
        datasets: [{
          label: t('hist_dataset_label'),
          data: counts.map(function (item) { return item.count; }),
          backgroundColor: counts.map(function (item) {
            return state.mapping[item.name] ? 'rgba(16, 185, 129, 0.75)' : 'rgba(148, 163, 184, 0.4)';
          }),
          borderRadius: 3
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false }, ticks: { color: '#94a3b8', font: { size: 9 } } },
          y: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8', font: { size: 9 } } }
        }
      }
    });
  }

  function renderMapping() {
    var body = el['mapping-body'];
    body.innerHTML = '';
    var countById = {};
    (state.report.class_counts || []).forEach(function (item) { countById[item.id] = item.count; });

    state.suggestions.forEach(function (item) {
      var row = document.createElement('tr');
      row.className = state.mapping[item.source_name] ? '' : 'opacity-60';

      var source = document.createElement('td');
      source.className = 'px-2 py-1.5 font-mono text-slate-200';
      source.textContent = item.source_name;

      var count = document.createElement('td');
      count.className = 'px-2 py-1.5 font-mono text-slate-400';
      count.textContent = countById[item.source_id] || 0;

      var target = document.createElement('td');
      target.className = 'px-2 py-1.5';
      var select = document.createElement('select');
      select.className = 'bg-slate-800 border border-slate-700 rounded-md px-1.5 py-1 text-xs text-slate-200 font-mono';
      var dropOption = document.createElement('option');
      dropOption.value = '';
      dropOption.textContent = t('drop_option');
      select.appendChild(dropOption);
      BALOOT_CLASSES.forEach(function (name) {
        var option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        select.appendChild(option);
      });
      select.value = state.mapping[item.source_name] || '';
      select.addEventListener('change', function () {
        state.mapping[item.source_name] = select.value || null;
        item.reason = 'manual';
        row.className = select.value ? '' : 'opacity-60';
        reason.textContent = reasonText(item.reason);
        renderMappingSummary();
        renderStatsFromMapping();
        renderHistogram();
      });
      target.appendChild(select);

      var reason = document.createElement('td');
      reason.className = 'px-2 py-1.5 text-[11px] text-slate-500';
      reason.textContent = reasonText(item.reason);

      row.append(source, count, target, reason);
      body.appendChild(row);
    });

    renderMappingSummary();
  }

  function renderStatsFromMapping() {
    var mapped = {};
    Object.keys(state.mapping).forEach(function (key) {
      if (state.mapping[key]) mapped[state.mapping[key]] = true;
    });
    el['stat-baloot'].textContent = Object.keys(mapped).length + ' / 32';
  }

  function renderMappingSummary() {
    var countById = {};
    (state.report.class_counts || []).forEach(function (item) { countById[item.id] = item.count; });
    var kept = 0;
    var dropped = 0;
    var boxes = 0;
    state.suggestions.forEach(function (item) {
      if (state.mapping[item.source_name]) {
        kept += 1;
        boxes += countById[item.source_id] || 0;
      } else {
        dropped += 1;
      }
    });
    el['mapping-summary'].textContent = t('map_summary', { kept: kept, boxes: boxes, dropped: dropped });
  }

  // ---- actions --------------------------------------------------------

  function inspectServerPath() {
    var path = el['ds-path'].value.trim();
    if (!path) return;
    if (!state.backend) {
      showStatus(t('need_backend_build') + ' uv run hakim-vision studio', 'warn');
      return;
    }
    state.source = path;
    showStatus(t('inspecting'), 'info');
    api('/api/dataset/inspect', { body: { path: path } })
      .then(function (payload) {
        adoptReport(payload.report, payload.suggestions, {});
      })
      .catch(function (error) {
        showStatus(t('error_prefix', { message: error.message }), 'error');
      });
  }

  function inspectPickedFolder(files) {
    if (!files || !files.length) return;
    showStatus(t('inspecting'), 'info');
    analyseFolder(files)
      .then(function (payload) {
        state.source = payload.report.root;
        adoptReport(payload.report, suggestLocally(payload.report.class_names), {
          truncated: payload.truncated
        });
      })
      .catch(function (error) {
        showStatus(t('error_prefix', { message: error.message }), 'error');
      });
  }

  function buildDataset() {
    if (!state.report) return;
    var output = el['ds-output'].value.trim();
    if (!output) return;
    if (!state.backend) {
      showStatus(t('need_backend_build') + ' uv run hakim-vision studio', 'warn');
      return;
    }
    showStatus(t('building'), 'info');
    api('/api/dataset/remap', {
      body: { path: state.source, output: output, mapping: state.mapping }
    })
      .then(function (payload) {
        watchJob(payload.job, function (job) {
          if (job.status === 'done') {
            el['train-data'].value = output.replace(/[\\/]+$/, '') + '/data.yaml';
            updateTrainCli();
            showStatus(t('build_done'), 'ok');
          } else if (job.status === 'failed') {
            showStatus(t('error_prefix', { message: job.tail || 'remap failed' }), 'error');
          }
        });
      })
      .catch(function (error) {
        showStatus(t('error_prefix', { message: error.message }), 'error');
      });
  }

  function trainingPayload() {
    return {
      data: el['train-data'].value.trim(),
      model: el['train-model'].value,
      epochs: parseInt(el['train-epochs'].value, 10) || 50,
      imgsz: parseInt(el['train-imgsz'].value, 10) || 640,
      batch: parseInt(el['train-batch'].value, 10) || 16,
      device: el['train-device'].value
    };
  }

  function updateTrainCli() {
    var config = trainingPayload();
    var parts = [
      'uv run hakim-vision train',
      '--data ' + (config.data || 'data/baloot-dataset/data.yaml'),
      '--model ' + config.model,
      '--epochs ' + config.epochs,
      '--imgsz ' + config.imgsz,
      '--batch ' + config.batch
    ];
    if (config.device) parts.push('--device ' + config.device);
    el['train-cli'].textContent = parts.join(' \\\n  ');
  }

  function startTraining() {
    var config = trainingPayload();
    if (!config.data) return;
    if (!state.backend) {
      setTrainStatus('idle', t('need_backend_train'));
      return;
    }
    if (!state.backend.ultralytics) {
      setTrainStatus('failed', t('need_ultralytics'));
      return;
    }
    el['train-log'].textContent = '';
    updateTrainCli();
    api('/api/train', { body: config })
      .then(function (payload) {
        watchJob(payload.job, null);
      })
      .catch(function (error) {
        setTrainStatus('failed', t('error_prefix', { message: error.message }));
      });
  }

  function stopTraining() {
    if (!state.job) return;
    api('/api/jobs/' + state.job.id + '/stop', { body: {} }).catch(function () { /* already gone */ });
  }

  function exportModel() {
    var weights = el['export-weights'].value.trim();
    if (!weights || !state.backend) return;
    el['export-log'].classList.remove('hidden');
    el['export-log'].textContent = t('exporting');
    api('/api/export', { body: { weights: weights, imgsz: parseInt(el['train-imgsz'].value, 10) || 640 } })
      .then(function (payload) {
        watchJob(payload.job, null);
      })
      .catch(function (error) {
        el['export-log'].textContent = t('error_prefix', { message: error.message });
      });
  }

  // ---- job polling ----------------------------------------------------

  function watchJob(job, onFinish) {
    state.job = job;
    if (state.pollTimer) clearInterval(state.pollTimer);

    var poll = function () {
      api('/api/jobs/' + job.id)
        .then(function (detail) {
          renderJob(detail);
          if (detail.status !== 'running') {
            clearInterval(state.pollTimer);
            state.pollTimer = null;
            if (onFinish) {
              onFinish({ status: detail.status, tail: (detail.log || []).slice(-1)[0] });
            }
          }
        })
        .catch(function () {
          clearInterval(state.pollTimer);
          state.pollTimer = null;
        });
    };

    poll();
    state.pollTimer = setInterval(poll, JOB_POLL_MS);
  }

  function renderJob(detail) {
    if (detail.kind === 'export') {
      el['export-log'].textContent = (detail.log || []).join('\n');
      return;
    }
    if (detail.kind === 'remap') {
      showStatus((detail.log || []).slice(-1)[0] || t('building'), detail.status === 'failed' ? 'error' : 'info');
      return;
    }

    el['train-log'].textContent = (detail.log || []).join('\n');
    el['train-log'].scrollTop = el['train-log'].scrollHeight;
    setTrainStatus(detail.status, null);

    var metrics = detail.metrics || [];
    if (metrics.length) {
      var last = metrics[metrics.length - 1];
      var total = parseInt(el['train-epochs'].value, 10) || metrics.length;
      el['train-epoch'].textContent = (last.epoch || metrics.length) + ' / ' + total;
      el['train-map'].textContent = last.map50 !== undefined ? (last.map50 * 100).toFixed(1) + '%' : '—';
      el['train-progress'].style.width = Math.min(100, ((last.epoch || metrics.length) / total) * 100) + '%';
      renderMetrics(metrics);
    }
  }

  function setTrainStatus(status, message) {
    var labels = {
      running: t('status_running'),
      done: t('status_done'),
      failed: t('status_failed'),
      stopped: t('status_stopped'),
      idle: t('status_idle')
    };
    el['train-status'].textContent = message || labels[status] || labels.idle;
    var running = status === 'running';
    el['btn-train-stop'].classList.toggle('hidden', !running);
    el['train-start-label'].textContent = running ? t('btn_training') : t('btn_train');
    el['btn-train-start'].disabled = running || !state.backend;
  }

  function renderMetrics(metrics) {
    var canvas = el['training-chart'];
    if (!canvas || typeof Chart === 'undefined') return;
    var labels = metrics.map(function (row, index) { return row.epoch || index + 1; });

    if (!state.chart) {
      state.chart = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: {
          labels: labels,
          datasets: [
            {
              label: t('chart_loss'),
              data: [],
              borderColor: '#f87171',
              backgroundColor: 'rgba(248, 113, 113, 0.1)',
              tension: 0.25,
              yAxisID: 'y'
            },
            {
              label: t('chart_map'),
              data: [],
              borderColor: '#34d399',
              backgroundColor: 'rgba(52, 211, 153, 0.1)',
              tension: 0.25,
              yAxisID: 'y1'
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          scales: {
            x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8', font: { size: 9 } } },
            y: { position: 'left', grid: { color: '#1e293b' }, ticks: { color: '#f87171', font: { size: 9 } } },
            y1: {
              position: 'right', min: 0, max: 1,
              grid: { drawOnChartArea: false },
              ticks: { color: '#34d399', font: { size: 9 } }
            }
          },
          plugins: { legend: { labels: { color: '#e2e8f0', font: { family: 'Tajawal, sans-serif', size: 11 } } } }
        }
      });
    }

    state.chart.data.labels = labels;
    state.chart.data.datasets[0].data = metrics.map(function (row) { return row.box_loss; });
    state.chart.data.datasets[1].data = metrics.map(function (row) { return row.map50; });
    state.chart.update();
  }

  // ---- wiring ---------------------------------------------------------

  function populateModels(models) {
    var select = el['train-model'];
    if (!select) return;
    var current = select.value;
    select.innerHTML = '';
    models.forEach(function (name) {
      var option = document.createElement('option');
      option.value = name;
      option.textContent = name;
      select.appendChild(option);
    });
    if (current) select.value = current;
    updateTrainCli();
  }

  function copyText(text, button) {
    if (!navigator.clipboard) return;
    navigator.clipboard.writeText(text).then(function () {
      var original = button.textContent;
      button.textContent = t('copied');
      setTimeout(function () { button.textContent = original; }, 1500);
    }, function () { /* clipboard blocked */ });
  }

  function cacheDom() {
    [
      'backend-badge', 'ds-path', 'btn-ds-inspect', 'ds-folder', 'ds-status', 'ds-report',
      'kaggle-cmd', 'btn-copy-kaggle', 'stat-images', 'stat-boxes', 'stat-classes', 'stat-baloot',
      'splits-body', 'issues-list', 'class-histogram', 'mapping-body', 'mapping-summary',
      'btn-map-suggest', 'btn-map-clear', 'ds-output', 'btn-ds-remap', 'remap-cli',
      'train-data', 'train-model', 'train-device', 'train-epochs', 'train-imgsz', 'train-batch',
      'train-status', 'train-epoch', 'train-map', 'train-progress', 'train-cli', 'train-log',
      'training-chart', 'btn-train-start', 'btn-train-stop', 'train-start-label',
      'btn-copy-train-cli', 'export-weights', 'btn-export', 'export-log'
    ].forEach(function (id) { el[id] = $(id); });
  }

  function bind() {
    el['btn-ds-inspect'].addEventListener('click', inspectServerPath);
    el['ds-path'].addEventListener('keydown', function (event) {
      if (event.key === 'Enter') inspectServerPath();
    });
    el['ds-folder'].addEventListener('change', function (event) {
      inspectPickedFolder(event.target.files);
    });
    el['btn-copy-kaggle'].addEventListener('click', function () {
      copyText(el['kaggle-cmd'].textContent, el['btn-copy-kaggle'].firstElementChild || el['btn-copy-kaggle']);
    });

    el['btn-map-suggest'].addEventListener('click', function () {
      state.suggestions.forEach(function (item) {
        var suggestion = normaliseCardName(item.source_name);
        state.mapping[item.source_name] =
          suggestion && BALOOT_CLASSES.indexOf(suggestion) !== -1 ? suggestion : null;
      });
      renderMapping();
      renderStatsFromMapping();
      renderHistogram();
    });
    el['btn-map-clear'].addEventListener('click', function () {
      Object.keys(state.mapping).forEach(function (key) { state.mapping[key] = null; });
      renderMapping();
      renderStatsFromMapping();
      renderHistogram();
    });

    el['btn-ds-remap'].addEventListener('click', buildDataset);
    el['btn-train-start'].addEventListener('click', startTraining);
    el['btn-train-stop'].addEventListener('click', stopTraining);
    el['btn-export'].addEventListener('click', exportModel);
    el['btn-copy-train-cli'].addEventListener('click', function () {
      copyText(el['train-cli'].textContent, el['btn-copy-train-cli'].firstElementChild || el['btn-copy-train-cli']);
    });

    ['train-data', 'train-model', 'train-device', 'train-epochs', 'train-imgsz', 'train-batch']
      .forEach(function (id) {
        el[id].addEventListener('change', updateTrainCli);
        el[id].addEventListener('input', updateTrainCli);
      });

    document.addEventListener('hakim:languagechange', function (event) {
      state.lang = event.detail && event.detail.lang === 'en' ? 'en' : 'ar';
      applyStrings();
      renderBackendBadge();
      if (state.report) {
        renderMapping();
        renderHistogram();
      }
    });
  }

  function init() {
    cacheDom();
    if (!el['btn-ds-inspect']) return; // not the studio page
    state.token = readToken();
    bind();
    applyStrings();
    updateTrainCli();
    setTrainStatus('idle', null);
    detectBackend();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Exposed for tests and for anyone scripting the studio from the console.
  window.HakimLab = {
    normaliseCardName: normaliseCardName,
    suggestLocally: suggestLocally,
    analyseFolder: analyseFolder
  };
})();
