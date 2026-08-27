/**
 * Hakim — Arabic-first bilingual strings.
 *
 * Arabic is the source language; English is the fallback locale. `t(key, vars)`
 * interpolates `{name}` placeholders.
 *
 * The interface is icon-first, so a large share of these strings are never
 * painted on screen: they are the accessible names of icon-only controls, read
 * by screen readers and shown as tooltips on pointer devices. An icon without
 * an `aria_*` entry here is an unlabelled button, so they are kept together
 * with the visible copy rather than inlined at the call site.
 */

var HakimI18N = (function () {
  'use strict';

  var STRINGS = {
    ar: {
      dir: 'rtl',
      lang_switch_label: 'English',
      skip_link: 'تخطَّ إلى المحتوى',
      app_name: 'حكيم',
      app_tagline: 'حاسبة البلوت',
      lab_link: 'مختبر البيانات',
      page_title: 'حكيم | حاسبة نقاط البلوت',

      // ---- Teams and the table ----------------------------------------
      team_us: 'لنا',
      team_them: 'لهم',
      team_us_default: 'لنا',
      team_them_default: 'لهم',
      round_points_hint: 'اضغط لإدخال النقاط',
      pending_zero: '—',
      target_label: 'اللعبة إلى {target}',
      leading: 'متقدّم',
      tied: 'تعادل',
      round_number: 'جولة {n}',
      game_live: 'جولة جارية',

      // ---- Icon-only controls (accessible names + tooltips) -----------
      aria_home: 'الرئيسية',
      aria_game: 'اللعبة',
      aria_history: 'السجل',
      aria_settings: 'الإعدادات',
      aria_scan: 'صوّر الورق واحسب',
      aria_calculate: 'احسب واعتمد الجولة',
      aria_undo: 'تراجع عن آخر جولة',
      aria_new_round: 'جولة جديدة',
      aria_new_game: 'لعبة جديدة',
      aria_stats: 'الإحصائيات',
      aria_more: 'خيارات أخرى',
      aria_close: 'إغلاق',
      aria_confirm: 'تأكيد',
      aria_edit: 'تعديل',
      aria_capture: 'التقط صورة',
      aria_gallery: 'اختر من الصور',
      aria_retake: 'صورة أخرى',
      aria_backspace: 'حذف رقم',
      aria_clear_entry: 'مسح المدخل',
      aria_points_for: 'نقاط {team}',
      aria_edit_card: 'عدّل {card}',
      aria_add_card: 'أضف ورقة',
      aria_language: 'تغيير اللغة',
      aria_lab: 'مختبر البيانات',
      aria_delete_round: 'احذف {round}',

      // ---- Home --------------------------------------------------------
      home_title: 'مجلسك',
      home_resume: 'أكمل اللعبة',
      home_start: 'ابدأ لعبة',
      home_no_game: 'ما في لعبة جارية. ابدأ وحدة جديدة.',
      home_games_won: 'الألعاب الملعوبة',
      home_rounds_played: 'الجولات',
      home_quick_scan: 'صوّر الورق',
      home_ai_hint: 'وجّه الكاميرا على الورق، وحكيم يحسب النقاط.',

      // ---- Keypad / Calculator -----------------------------------------
      keypad_title: 'نقاط {team}',
      keypad_hint: 'اكتب نقاط هذه الجولة أو احسبها',
      calc_title: 'حاسبة نقاط الجولة',
      calc_camera_ai: 'صوّر بالذكاء الاصطناعي',
      calc_camera_ai_hint: 'امسح الورق بالكاميرا لحساب النقاط آلياً',
      calc_edit_ai_hint: 'اضغط أي ورقة لتعديلها أو أضف ورقة جديدة',
      calc_projects_title: 'المشاريع',
      calc_base_points: 'نقاط اللعب',
      calc_project_points: 'المشاريع',
      calc_total_points: 'المجموع الكلي',
      calc_clear_projects: 'مسح المشاريع',
      calc_clear_all: 'مسح',
      calc_constant_label: 'مجموع اللعب: {total}',
      calc_auto_calc_hint: 'حساب تلقائي متكامل: مجموع نقاط اللعب الصافية ثابت {total}',
      calc_team_playing: 'لعب',
      calc_team_projects: 'مشاريع',
      btn_save_points: 'اعتمد',
      btn_calc_confirm: 'اعتمد الجولة لكلا الفريقين',

      // ---- Calculate ---------------------------------------------------
      btn_calculate: 'احسب',
      btn_new_round: 'قيد جديد',
      btn_undo: 'تراجع',
      toast_round_added: 'انضافت الجولة',
      toast_undone: 'رجعنا خطوة',
      toast_nothing_undo: 'ما في جولة نرجعها',
      toast_new_game: 'لعبة جديدة',
      toast_reset: 'انمسح كل شيء',
      toast_empty_round: 'أدخل نقاط أول',
      toast_hand_ready: 'تم اعتماد الأوراق',
      toast_copied: 'تم النسخ',

      // ---- Win ---------------------------------------------------------
      win_title: '{team} كسبت',
      win_score: '{us} — {them}',
      win_new_game: 'لعبة جديدة',
      win_dismiss: 'كمّل',

      // ---- Scanner -----------------------------------------------------
      scanner_title: 'ماسح الأوراق',
      scan_aim: 'وجّه الكاميرا على الورق',
      scan_aim_hint: 'افرد الأوراق على سطح واضح وصوّرها من الأعلى.',
      scan_step_detect: 'نلقط الورق',
      scan_step_read: 'نقرأ القيم',
      scan_step_calc: 'نحسب النقاط',
      scan_manual: 'يدويًا',
      privacy_note: 'الصورة تُعالَج داخل جهازك ولا تُرفع لأي مكان.',
      detect_none: 'ما لقينا ورق. جرّب إضاءة أوضح وخلفية داكنة، أو أضف الأوراق يدويًا.',
      detect_needs_naming: 'اضغط أي موضع لتحديد ورقته.',
      detect_error: 'تعذّر تحليل الصورة. تقدر تدخل الأوراق يدويًا.',
      detect_diagnostics: '{backend} ({variant}) • {ms} مللي ثانية',

      // ---- Verification ------------------------------------------------
      verify_question: 'هل النتيجة صحيحة؟',
      verify_hint: 'اضغط أي ورقة لتصحيحها.',
      verify_confirm: 'صحيحة',
      verify_edit: 'عدّل',
      verify_assign: 'النقاط لِـ',
      hand_value: 'قيمة اليد',

      // ---- Hand review -------------------------------------------------
      hand_title: 'راجع الأوراق',
      hand_count: '{count} / {expected}',
      btn_clear_hand: 'مسح الكل',
      btn_copy_result: 'انسخ النتيجة',
      add_card: 'أضف ورقة',
      empty_hand: 'ما في أوراق بعد.',
      unknown_card: 'ورقة غير محدّدة',
      tap_to_identify: 'حدّدها',
      picker_title: 'أي ورقة هذه؟',
      btn_remove_card: 'احذف هذه الورقة',

      notice_duplicate: 'الورقة {card} مكررة — ما تتكرر في يد وحدة.',
      notice_unknown: '{count} ورقة تحتاج تحديد.',
      notice_hand_size: 'يدك فيها {count} ورقة، ويد البلوت ٨ أوراق.',
      notice_no_trump: 'اختر الحكم عشان تنحسب قيم الجوكر والتسعة.',

      // ---- Declaration -------------------------------------------------
      mode_sun: 'صن',
      mode_hokum: 'حكم',
      mode_group_label: 'نوع اللعب',
      lbl_trump: 'اختر الحكم',
      lbl_total: 'مجموع النقاط',
      lbl_card_points: 'نقاط الأوراق',
      lbl_project_points: 'نقاط المشاريع',
      lbl_breakdown: 'تفصيل النقاط ورقة بورقة',
      th_card: 'الورقة',
      th_note: 'ملاحظة',
      th_points: 'النقاط',
      share_text: '{points} من {total} نقطة في الصكة.',
      note_trump: 'حكم',
      suggestion_best: 'أعلى قيمة ليدك: {label} بمجموع {total}.',
      suggestion_current: 'هذا الاختيار هو الأعلى قيمة ليدك.',

      // ---- History -----------------------------------------------------
      history_title: 'السجل',
      history_empty: 'ما في جولات بعد. أول قيد يبدأ من هنا.',
      history_ai_badge: 'محسوبة من صورة',
      history_running: 'المجموع',

      // ---- Statistics --------------------------------------------------
      stats_title: 'الإحصائيات',
      stat_rounds: 'الجولات',
      stat_best: 'أعلى جولة',
      stat_rounds_won: 'جولات مكسوبة',
      stat_average: 'متوسط الجولة',
      stat_games: 'الألعاب',
      stat_remaining: 'الباقي للفوز',
      stat_scanned: 'جولات بالكاميرا',

      // ---- Settings ----------------------------------------------------
      settings_title: 'الإعدادات',
      set_names: 'أسماء الفرق',
      set_name_us: 'اسم فريقنا',
      set_name_them: 'اسم فريقهم',
      set_target: 'اللعبة إلى',
      set_language: 'اللغة',
      set_game: 'اللعبة',
      set_new_game: 'لعبة جديدة',
      set_reset: 'امسح كل شيء',
      set_reset_hint: 'يمسح النتائج والأسماء والألعاب المكتسبة.',
      set_about: 'عن حكيم',
      rules_note: 'القيم المعتمدة: صن ١٢٠ نقطة، حكم ١٥٢ نقطة (دون العشرة الأخيرة). المشاريع: سرا ٢٠، خمسين ٥٠، مية ١٠٠، أربعمية ٤٠٠، بلوت ٢٠ في الحكم. قواعد المشاريع تختلف بين المجالس.',
      footer_text: 'حكيم • مشروع مفتوح المصدر للرؤية الحاسوبية في البلوت السعودي • ترخيص MIT',

      // ---- Projects and suits -----------------------------------------
      project_sra: 'سرا',
      project_fifty: 'خمسين',
      project_hundred: 'مية',
      project_fourHundred: 'أربعمية',
      project_baloot: 'بلوت',
      suit_h: 'هاص',
      suit_d: 'ديمن',
      suit_c: 'كلوب',
      suit_s: 'سبيت'
    },

    en: {
      dir: 'ltr',
      lang_switch_label: 'العربية',
      skip_link: 'Skip to content',
      app_name: 'Hakim',
      app_tagline: 'Baloot scorer',
      lab_link: 'Dataset lab',
      page_title: 'Hakim | Baloot score calculator',

      team_us: 'Us',
      team_them: 'Them',
      team_us_default: 'Us',
      team_them_default: 'Them',
      round_points_hint: 'Tap to enter points',
      pending_zero: '—',
      target_label: 'Playing to {target}',
      leading: 'Leading',
      tied: 'Tied',
      round_number: 'Round {n}',
      game_live: 'Game in play',

      aria_home: 'Home',
      aria_game: 'Game',
      aria_history: 'History',
      aria_settings: 'Settings',
      aria_scan: 'Scan the cards and score',
      aria_calculate: 'Calculate and record the round',
      aria_undo: 'Undo the last round',
      aria_new_round: 'New round',
      aria_new_game: 'New game',
      aria_stats: 'Statistics',
      aria_more: 'More options',
      aria_close: 'Close',
      aria_confirm: 'Confirm',
      aria_edit: 'Edit',
      aria_capture: 'Take a photo',
      aria_gallery: 'Choose from photos',
      aria_retake: 'Retake',
      aria_backspace: 'Delete a digit',
      aria_clear_entry: 'Clear the entry',
      aria_points_for: 'Points for {team}',
      aria_edit_card: 'Edit {card}',
      aria_add_card: 'Add a card',
      aria_language: 'Change language',
      aria_lab: 'Dataset lab',
      aria_delete_round: 'Delete {round}',

      home_title: 'Your table',
      home_resume: 'Resume game',
      home_start: 'Start a game',
      home_no_game: 'No game in play. Start a new one.',
      home_games_won: 'Games won',
      home_rounds_played: 'Rounds',
      home_quick_scan: 'Scan the cards',
      home_ai_hint: 'Point the camera at the cards and Hakim does the maths.',

      // ---- Keypad / Calculator -----------------------------------------
      keypad_title: 'Points for {team}',
      keypad_hint: 'Enter or calculate this round’s points',
      calc_title: 'Round Score Calculator',
      calc_camera_ai: 'Scan with AI',
      calc_camera_ai_hint: 'Scan cards with camera for instant AI calculation',
      calc_edit_ai_hint: 'Tap any card to edit it or add a new card',
      calc_projects_title: 'Projects',
      calc_base_points: 'Game Points',
      calc_project_points: 'Projects',
      calc_total_points: 'Total Score',
      calc_clear_projects: 'Clear projects',
      calc_clear_all: 'Clear',
      calc_constant_label: 'Deck trick total: {total}',
      calc_auto_calc_hint: 'Auto-balanced: base trick points total is constant {total}',
      calc_team_playing: 'Game',
      calc_team_projects: 'Projects',
      btn_save_points: 'Save',
      btn_calc_confirm: 'Commit Round for Both Teams',

      btn_calculate: 'Score',
      btn_new_round: 'New round',
      btn_undo: 'Undo',
      toast_round_added: 'Round recorded',
      toast_undone: 'Stepped back',
      toast_nothing_undo: 'Nothing to undo',
      toast_new_game: 'New game',
      toast_reset: 'Everything cleared',
      toast_empty_round: 'Enter some points first',
      toast_hand_ready: 'Cards accepted',
      toast_copied: 'Copied',

      win_title: '{team} win',
      win_score: '{us} — {them}',
      win_new_game: 'New game',
      win_dismiss: 'Keep playing',

      scanner_title: 'Card scanner',
      scan_aim: 'Point the camera at the cards',
      scan_aim_hint: 'Spread the cards on a clear surface and shoot from above.',
      scan_step_detect: 'Finding cards',
      scan_step_read: 'Reading values',
      scan_step_calc: 'Working out points',
      scan_manual: 'By hand',
      privacy_note: 'The photo is processed on your device and never uploaded.',
      detect_none: 'No cards found. Try brighter light and a darker background, or add them by hand.',
      detect_needs_naming: 'Tap a position to name its card.',
      detect_error: 'The photo could not be analysed. You can enter the cards by hand.',
      detect_diagnostics: '{backend} ({variant}) • {ms} ms',

      verify_question: 'Is this right?',
      verify_hint: 'Tap any card to correct it.',
      verify_confirm: 'Correct',
      verify_edit: 'Edit',
      verify_assign: 'Points go to',
      hand_value: 'Hand value',

      hand_title: 'Review the cards',
      hand_count: '{count} / {expected}',
      btn_clear_hand: 'Clear all',
      btn_copy_result: 'Copy result',
      add_card: 'Add card',
      empty_hand: 'No cards yet.',
      unknown_card: 'Unidentified card',
      tap_to_identify: 'Identify',
      picker_title: 'Which card is this?',
      btn_remove_card: 'Remove this card',

      notice_duplicate: '{card} appears twice — impossible in one hand.',
      notice_unknown: '{count} card(s) still need identifying.',
      notice_hand_size: 'You have {count} cards; a full Baloot hand is 8.',
      notice_no_trump: 'Pick the trump suit to value the jack and nine correctly.',

      mode_sun: 'Sun',
      mode_hokum: 'Hokum',
      mode_group_label: 'Declaration',
      lbl_trump: 'Pick the trump suit',
      lbl_total: 'Total points',
      lbl_card_points: 'Card points',
      lbl_project_points: 'Project points',
      lbl_breakdown: 'Point-by-point breakdown',
      th_card: 'Card',
      th_note: 'Note',
      th_points: 'Points',
      share_text: '{points} of the {total} card points in the deck.',
      note_trump: 'trump',
      suggestion_best: 'Best declaration: {label}, worth {total}.',
      suggestion_current: 'This is already the best declaration for your hand.',

      history_title: 'History',
      history_empty: 'No rounds yet. The first one starts here.',
      history_ai_badge: 'Scored from a photo',
      history_running: 'Total',

      stats_title: 'Statistics',
      stat_rounds: 'Rounds',
      stat_best: 'Best round',
      stat_rounds_won: 'Rounds won',
      stat_average: 'Average round',
      stat_games: 'Games',
      stat_remaining: 'Left to win',
      stat_scanned: 'Scanned rounds',

      settings_title: 'Settings',
      set_names: 'Team names',
      set_name_us: 'Our team',
      set_name_them: 'Their team',
      set_target: 'Playing to',
      set_language: 'Language',
      set_game: 'Game',
      set_new_game: 'New game',
      set_reset: 'Clear everything',
      set_reset_hint: 'Wipes scores, names, and games won.',
      set_about: 'About Hakim',
      rules_note: 'Values used: Sun 120 points, Hokum 152 points (last trick excluded). Projects: sra 20, fifty 50, hundred 100, four hundred 400, baloot 20 in Hokum. Project rules vary between houses.',
      footer_text: 'Hakim • open-source computer vision for Saudi Baloot • MIT licence',

      project_sra: 'Sra',
      project_fifty: 'Fifty',
      project_hundred: 'Hundred',
      project_fourHundred: 'Four hundred',
      project_baloot: 'Baloot',
      suit_h: 'Hearts',
      suit_d: 'Diamonds',
      suit_c: 'Clubs',
      suit_s: 'Spades'
    }
  };

  var current = 'ar';

  function setLanguage(lang) {
    current = STRINGS[lang] ? lang : 'ar';
    return current;
  }

  function getLanguage() {
    return current;
  }

  function t(key, vars) {
    var dict = STRINGS[current] || STRINGS.ar;
    var value = dict[key];
    if (value === undefined) value = STRINGS.ar[key];
    if (value === undefined) return key;
    if (!vars) return value;
    return value.replace(/\{(\w+)\}/g, function (match, name) {
      return Object.prototype.hasOwnProperty.call(vars, name) ? String(vars[name]) : match;
    });
  }

  /**
   * Localise the document.
   *
   * Three passes, because an icon-first UI keeps its copy in three places:
   * `data-i18n` for visible text, `data-i18n-aria-label` for the accessible
   * name of an icon-only control, and `data-i18n-title` for the tooltip that
   * gives a pointer user the same information. Without the latter two, every
   * icon button would stay frozen in whichever language the page first loaded.
   */
  function applyToDocument(root) {
    var scope = root || document;
    scope.querySelectorAll('[data-i18n]').forEach(function (element) {
      element.textContent = t(element.getAttribute('data-i18n'));
    });
    scope.querySelectorAll('[data-i18n-aria-label]').forEach(function (element) {
      element.setAttribute('aria-label', t(element.getAttribute('data-i18n-aria-label')));
    });
    scope.querySelectorAll('[data-i18n-title]').forEach(function (element) {
      element.setAttribute('title', t(element.getAttribute('data-i18n-title')));
    });
  }

  return {
    STRINGS: STRINGS,
    setLanguage: setLanguage,
    getLanguage: getLanguage,
    t: t,
    applyToDocument: applyToDocument
  };
})();

if (typeof module !== 'undefined' && module.exports) {
  module.exports = HakimI18N;
}
