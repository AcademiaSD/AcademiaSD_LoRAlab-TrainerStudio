// i18n.js — Traducciones de la interfaz / Interface translations.
// Las páginas se escriben en inglés y la clave de cada traducción es el propio texto inglés
// (como en gettext): GUI/locales/<idioma>.json = {"Save": "Guardar", ...}. Lo que no esté traducido
// sale en inglés, así que un texto nuevo funciona antes de traducirlo.
//   I18N.apply(raíz)        traduce los textos y los atributos title / placeholder bajo la raíz
//                           (los ":" del final y los espacios no forman parte de la clave)
//   t('Hello {name}', {name: 'x'})   desde el JS; {name} se sustituye
//   class="no-i18n"         no traduce ese elemento ni lo de dentro (captions, consola, nombres...)
// await I18N.load() antes de pintar la página; I18N.load('de') carga otro idioma sin guardarlo.
const I18N = {
    lang: 'en',
    languages: [],   // [[código, nombre], ...] en el orden del selector
    strings: {},
    original: new WeakMap(),   // texto inglés de cada nodo, para poder cambiar de idioma otra vez

    async load(lang) {
        try {
            const res = await fetch('/api/i18n' + (lang ? '?lang=' + encodeURIComponent(lang) : ''));
            const data = await res.json();
            Object.assign(this, { lang: data.language, languages: data.languages, strings: data.strings });
        } catch (e) {
            console.warn('i18n:', e);
        }
        document.documentElement.lang = this.lang;
        this.apply(document.body);
    },

    // "  Project Name:  " -> se traduce "Project Name" y se conservan los espacios y los ":".
    translate(text) {
        const m = /^(\s*)([\s\S]*?)(\s*:?\s*)$/.exec(text);
        const key = m[2].replace(/\s+/g, ' ');
        return key && key in this.strings ? m[1] + this.strings[key] + m[3] : text;
    },

    apply(root) {
        const skip = el => el.closest('.no-i18n');
        const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
        for (let node; (node = walker.nextNode());) {
            if (!node.nodeValue.trim() || node.parentElement.closest('.no-i18n, script, style, textarea')) continue;
            if (!this.original.has(node)) this.original.set(node, node.nodeValue);
            node.nodeValue = this.translate(this.original.get(node));
        }
        for (const attr of ['title', 'placeholder']) {
            root.querySelectorAll(`[${attr}]`).forEach(el => {
                if (skip(el)) return;
                const store = 'i18n' + attr;
                if (!(store in el.dataset)) el.dataset[store] = el.getAttribute(attr);
                el.setAttribute(attr, this.translate(el.dataset[store]));
            });
        }
    }
};

function t(key, vars) {
    let text = I18N.strings[key] ?? key;
    if (vars) text = text.replace(/\{(\w+)\}/g, (m, k) => (k in vars ? vars[k] : m));
    return text;
}
