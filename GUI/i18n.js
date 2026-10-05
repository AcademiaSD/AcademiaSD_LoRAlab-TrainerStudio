// i18n.js — Traducciones de la interfaz / Interface translations.
// Los textos vienen de /api/i18n (GUI/locales/<idioma>.json, con el inglés de reserva).
//   <span data-i18n="clave">            textContent
//   <span data-i18n-html="clave">       innerHTML (textos con <b>, <br>...)
//   data-i18n-title / data-i18n-placeholder / data-i18n-alt   atributos
//   t('clave', {name: 'x'})             desde el JS; {name} en el texto se sustituye
// await I18N.load() antes de pintar la página; I18N.load('de') carga otro idioma sin guardarlo.
const I18N = {
    lang: 'en',
    languages: [],   // [[código, nombre], ...] en el orden del selector
    strings: {},

    async load(lang) {
        try {
            const res = await fetch('/api/i18n' + (lang ? '?lang=' + encodeURIComponent(lang) : ''));
            const data = await res.json();
            Object.assign(this, { lang: data.language, languages: data.languages, strings: data.strings });
        } catch (e) {
            console.warn('i18n:', e);
        }
        document.documentElement.lang = this.lang;
        this.apply(document);
    },

    apply(root) {
        root.querySelectorAll('[data-i18n]').forEach(el => { el.textContent = t(el.dataset.i18n); });
        root.querySelectorAll('[data-i18n-html]').forEach(el => { el.innerHTML = t(el.dataset.i18nHtml); });
        for (const attr of ['title', 'placeholder', 'alt']) {
            const key = 'i18n' + attr[0].toUpperCase() + attr.slice(1);
            root.querySelectorAll(`[data-i18n-${attr}]`).forEach(el => { el.setAttribute(attr, t(el.dataset[key])); });
        }
    }
};

function t(key, vars) {
    let text = I18N.strings[key] ?? key;
    if (vars) text = text.replace(/\{(\w+)\}/g, (m, k) => (k in vars ? vars[k] : m));
    return text;
}
