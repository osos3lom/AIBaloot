/**
 * Hakim — Arabic-first bilingual strings.
 *
 * Arabic is the source language; English is the fallback locale. `t(key, vars)`
 * interpolates `{name}` placeholders.
 */

var HakimI18N = (function () {
  'use strict';

  var STRINGS = {
    ar: {
      dir: 'rtl',
      lang_switch_label: 'English',
      skip_link: 'تخطَّ إلى المحتوى',
      app_name: 'حكيم',
      app_tagline: 'احسب قيمة يدك',
      lab_link: 'مختبر البيانات',
      page_title: 'حكيم | احسب قيمة يدك في البلوت',

      step1_title: 'صوّر الأوراق التي بيدك',
      capture_hint: 'افرد أوراقك على سطح واضح وصوّرها من الأعلى، أو ارفع صورة جاهزة.',
      btn_camera: 'التقط صورة',
      btn_upload: 'اختر من الجهاز',
      privacy_note: 'الصورة تُعالَج داخل متصفحك ولا تُرفع إلى أي خادم.',
      btn_skip_photo: 'أو أدخل الأوراق يدويًا بدون صورة',
      detecting: 'جارٍ تحليل الصورة…',
      btn_redetect: 'أعد التحليل',
      btn_retake: 'صورة أخرى',
      detect_found: 'حدّد حكيم {count} موضع ورقة في {ms} مللي ثانية — راجعها وأضف ما فاته.',
      detect_none: 'لم يرصد حكيم أي ورقة. جرّب إضاءة أوضح وخلفية داكنة، أو أضف الأوراق يدويًا.',
      detect_needs_naming: 'نموذج التعرّف على القيم غير محمَّل بعد، لذلك اضغط كل موضع لتحديد ورقته.',
      detect_error: 'تعذّر تحليل الصورة. يمكنك إدخال الأوراق يدويًا.',

      step2_title: 'راجع الأوراق',
      step2_hint: 'اضغط أي ورقة لتصحيحها، والعلامة ؟ تعني موضعًا رصده حكيم ولم يحدّد قيمته. أضف أي ورقة فاتته بزر +.',
      hand_count: '{count} / {expected}',
      btn_clear_hand: 'مسح الكل',
      add_card: 'أضف ورقة',
      empty_hand: 'لا توجد أوراق بعد. صوّر يدك أو أضف الأوراق واحدة واحدة.',
      unknown_card: 'ورقة غير محدّدة',
      tap_to_identify: 'حدّدها',
      remove_card_label: 'احذف {card}',
      edit_card_label: 'عدّل {card}',

      notice_duplicate: 'الورقة {card} مكررة — لا يمكن أن تتكرر في يد واحدة.',
      notice_unknown: '{count} ورقة بحاجة إلى تحديد قبل اكتمال الحساب.',
      notice_hand_size: 'يدك تحتوي {count} ورقة، ويد البلوت الكاملة ٨ أوراق.',
      notice_no_trump: 'اختر الحكم لحساب قيم الجوكر والتسعة.',

      step3_title: 'قيمة يدك',
      mode_sun: 'صن',
      mode_hokum: 'حكم',
      lbl_trump: 'اختر الحكم',
      lbl_total: 'مجموع النقاط',
      lbl_card_points: 'نقاط الأوراق',
      lbl_project_points: 'نقاط المشاريع',
      lbl_breakdown: 'تفصيل النقاط ورقة بورقة',
      th_card: 'الورقة',
      th_note: 'ملاحظة',
      th_points: 'النقاط',
      share_text: '{points} من {total} نقطة موجودة في الصكة.',
      note_trump: 'حكم',
      btn_copy_result: 'انسخ النتيجة',
      copied: 'تم النسخ',
      suggestion_best: 'أعلى قيمة ليدك: {label} بمجموع {total} نقطة.',
      suggestion_current: 'هذا الاختيار هو الأعلى قيمة ليدك حاليًا.',
      rules_note: 'القيم المعتمدة: صن ١٢٠ نقطة، حكم ١٥٢ نقطة (دون العشرة الأخيرة). المشاريع: سرا ٢٠، خمسين ٥٠، مية ١٠٠، أربعمية ٤٠٠، بلوت ٢٠ في الحكم. قواعد المشاريع تختلف بين المجالس.',

      sticky_label_sun: 'صن',
      sticky_label_hokum: 'حكم {suit}',
      sticky_action: 'أضف ورقة',

      picker_title: 'أي ورقة هذه؟',
      btn_close: 'إغلاق',
      btn_remove_card: 'احذف هذه الورقة',

      project_sra: 'سرا',
      project_fifty: 'خمسين',
      project_hundred: 'مية',
      project_fourHundred: 'أربعمية',
      project_baloot: 'بلوت',

      suit_h: 'هاص',
      suit_d: 'ديمن',
      suit_c: 'كلوب',
      suit_s: 'سبيت',

      footer_text: 'حكيم • مشروع مفتوح المصدر للرؤية الحاسوبية في البلوت السعودي • ترخيص MIT'
    },

    en: {
      dir: 'ltr',
      lang_switch_label: 'العربية',
      skip_link: 'Skip to content',
      app_name: 'Hakim',
      app_tagline: 'Score your hand',
      lab_link: 'Dataset lab',
      page_title: 'Hakim | Score your Baloot hand',

      step1_title: 'Photograph the cards in your hand',
      capture_hint: 'Spread your cards on a clear surface and shoot from above, or upload a photo you already have.',
      btn_camera: 'Take a photo',
      btn_upload: 'Choose a file',
      privacy_note: 'The photo is processed inside your browser and never uploaded.',
      btn_skip_photo: 'Or enter the cards by hand, no photo needed',
      detecting: 'Analysing the photo…',
      btn_redetect: 'Analyse again',
      btn_retake: 'Different photo',
      detect_found: 'Hakim proposed {count} card positions in {ms} ms — review them and add any it missed.',
      detect_none: 'No cards found. Try brighter light and a darker background, or add the cards manually.',
      detect_needs_naming: 'No rank/suit model is loaded yet, so tap each position to name its card.',
      detect_error: 'The photo could not be analysed. You can enter the cards manually.',

      step2_title: 'Review the cards',
      step2_hint: 'Tap any card to correct it. A ? is a position Hakim found but could not name. Use + for cards it missed.',
      hand_count: '{count} / {expected}',
      btn_clear_hand: 'Clear all',
      add_card: 'Add card',
      empty_hand: 'No cards yet. Photograph your hand or add cards one by one.',
      unknown_card: 'Unidentified card',
      tap_to_identify: 'Identify',
      remove_card_label: 'Remove {card}',
      edit_card_label: 'Edit {card}',

      notice_duplicate: '{card} appears twice — impossible in one hand.',
      notice_unknown: '{count} card(s) still need identifying before the total is complete.',
      notice_hand_size: 'You have {count} cards; a full Baloot hand is 8.',
      notice_no_trump: 'Pick the trump suit to value the jack and nine correctly.',

      step3_title: 'Your hand value',
      mode_sun: 'Sun',
      mode_hokum: 'Hokum',
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
      btn_copy_result: 'Copy result',
      copied: 'Copied',
      suggestion_best: 'Best declaration for this hand: {label}, worth {total}.',
      suggestion_current: 'This is already the best declaration for your hand.',
      rules_note: 'Values used: Sun 120 points, Hokum 152 points (last trick excluded). Projects: sra 20, fifty 50, hundred 100, four hundred 400, baloot 20 in Hokum. Project rules vary between houses.',

      sticky_label_sun: 'Sun',
      sticky_label_hokum: 'Hokum {suit}',
      sticky_action: 'Add card',

      picker_title: 'Which card is this?',
      btn_close: 'Close',
      btn_remove_card: 'Remove this card',

      project_sra: 'Sra',
      project_fifty: 'Fifty',
      project_hundred: 'Hundred',
      project_fourHundred: 'Four hundred',
      project_baloot: 'Baloot',

      suit_h: 'Hearts',
      suit_d: 'Diamonds',
      suit_c: 'Clubs',
      suit_s: 'Spades',

      footer_text: 'Hakim • open-source computer vision for Saudi Baloot • MIT licence'
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

  /** Replace the text of every `[data-i18n]` element with the current locale. */
  function applyToDocument(root) {
    var scope = root || document;
    scope.querySelectorAll('[data-i18n]').forEach(function (element) {
      element.textContent = t(element.getAttribute('data-i18n'));
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
