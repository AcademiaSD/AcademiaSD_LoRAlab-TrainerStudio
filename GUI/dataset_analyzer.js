// dataset_analyzer.js — Botón «🔍 Analyze» en la barra del Dataset Manager de cada entrenador.
// Dataset analyzer button in the Dataset Manager toolbar of every trainer.
//
// Pide el análisis a /api/dataset-analyze (scripts/dataset_analyzer.py) y lo enseña en una ventana dentro de
// la página: un resumen (imágenes, captions, proporciones, caras) y los avisos, cada uno con las miniaturas de
// sus imágenes. Un clic en una miniatura abre esa imagen en el editor de captions del Dataset Manager.
(function () {
    const tr = (k, v) => (typeof t === 'function' ? t(k, v) : k.replace(/\{(\w+)\}/g, (m, n) => (v && n in v ? v[n] : m)));
    const $ = (id) => document.getElementById(id);
    const esc = (s) => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    let box = null, last = null, fromReport = false;

    // El último informe de cada carpeta de dataset se guarda en el navegador: 📋 lo vuelve a abrir sin repetir el
    // análisis, también tras recargar la página.
    const storeKey = () => 'loralab_dataset_analysis:' + (($('dataset_path')?.value || '').trim() || '-');
    function saveLast(data) {
        try {
            const copy = Object.assign({}, data);
            delete copy.captions_text;           // los captions se leen al día del Dataset Manager
            localStorage.setItem(storeKey(), JSON.stringify(Object.assign(copy, { when: Date.now() })));
        } catch (e) {}
    }
    function loadLast() {
        try { return JSON.parse(localStorage.getItem(storeKey()) || 'null'); } catch (e) { return null; }
    }
    function updateReopen() {
        const b = $('btn-ds-analysis-last');
        if (!b) return;
        const d = loadLast();
        b.style.display = d && !d.error ? '' : 'none';
        if (d && !d.error) {
            const warns = (d.issues || []).filter(i => i.level === 'warn').length;
            b.textContent = '📋' + (warns ? ' ' + warns : '');
            b.title = tr('Last analysis of this dataset ({when}): open it again without analyzing.',
                         { when: new Date(d.when).toLocaleString() });
        }
    }

    const CSS = `
    .da-box { position: fixed; inset: 0; background: rgba(2,6,23,.82); z-index: 2500; display: none; align-items: center; justify-content: center; padding: 20px; }
    .da-win { background: var(--panel-bg, #0f172a); border: 1px solid var(--panel-border, #334155); border-radius: 12px; width: min(980px, 100%);
              max-height: 90vh; display: flex; flex-direction: column; }
    .da-head { display: flex; align-items: center; gap: 10px; padding: 14px 18px; border-bottom: 1px solid var(--panel-border, #334155); }
    .da-head h3 { margin: 0; font-size: 1.05rem; flex: 1; }
    .da-head button { padding: 6px 12px; border-radius: 6px; border: 1px solid #334155; background: #1e293b; color: #e2e8f0; cursor: pointer; }
    .da-body { overflow: auto; padding: 14px 18px; }
    .da-sum { display: flex; flex-wrap: wrap; gap: 6px 8px; margin-bottom: 12px; }
    .da-chip { font-size: .78rem; padding: 3px 9px; border-radius: 999px; background: #1e293b; color: #cbd5e1; border: 1px solid #334155; }
    .da-ok { padding: 14px; border-radius: 8px; background: rgba(16,185,129,.1); border: 1px solid rgba(16,185,129,.4); color: #a7f3d0; }
    .da-issue { border: 1px solid #334155; border-left-width: 4px; border-radius: 8px; padding: 10px 12px; margin-bottom: 10px; background: #0b1222; }
    .da-issue.warn { border-left-color: #f59e0b; }
    .da-issue.info { border-left-color: #38bdf8; }
    .da-msg { font-size: .86rem; color: #e2e8f0; line-height: 1.45; }
    .da-msg b { color: #fbbf24; }
    .da-issue.info .da-msg b { color: #7dd3fc; }
    .da-thumbs { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
    .da-thumb { position: relative; cursor: pointer; }
    .da-thumb img { height: 64px; border-radius: 5px; border: 1px solid #334155; display: block; }
    .da-thumb:hover img { border-color: #38bdf8; }
    .da-thumb span { position: absolute; left: 2px; bottom: 2px; font-size: .62rem; background: rgba(0,0,0,.7); color: #fff; padding: 0 4px; border-radius: 3px; }
    .da-more { font-size: .75rem; color: #94a3b8; align-self: center; }
    .da-wait { color: #94a3b8; font-size: .9rem; padding: 20px 4px; }
    .da-when { font-size: .75rem; color: #94a3b8; margin-bottom: 8px; }
    .da-x { position: absolute; top: 2px; right: 2px; width: 20px; height: 20px; border-radius: 50%; border: 1px solid #475569;
            background: rgba(15,23,42,.85); color: #94a3b8; font-size: 11px; line-height: 1; cursor: pointer; padding: 0; }
    .da-x.on { background: #dc2626; border-color: #fca5a5; color: #fff; }
    .da-thumb:has(.da-x.on) img { opacity: .45; outline: 2px solid #dc2626; }
    .da-act { margin-top: 8px; display: flex; flex-direction: column; gap: 4px; font-size: .82rem; color: #cbd5e1; }
    .da-act label { display: flex; align-items: center; gap: 8px; cursor: pointer; }
    .da-act input { width: 15px; height: 15px; }
    .da-btns { display: flex; gap: 8px; margin-top: 4px; flex-wrap: wrap; }
    .da-btns button { padding: 5px 10px; border-radius: 6px; border: 1px solid #334155; background: #1e293b; color: #e2e8f0; cursor: pointer; font-size: .8rem; }
    .da-foot { display: flex; align-items: center; gap: 12px; padding: 10px 18px; border-top: 1px solid var(--panel-border, #334155); }
    .da-sel { flex: 1; font-size: .82rem; color: #94a3b8; }
    .da-apply { padding: 8px 16px; border-radius: 8px; border: none; font-weight: 600; cursor: pointer;
                background: linear-gradient(90deg, #059669, #2563eb); color: #fff; }
    .da-apply:disabled { opacity: .4; cursor: not-allowed; }`;

    function build() {
        const bar = document.querySelector('.ds-toolbar');
        if (!bar || $('btn-ds-analyze')) return;
        const style = document.createElement('style');
        style.textContent = CSS;
        document.head.appendChild(style);
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.id = 'btn-ds-analyze';
        btn.className = 'btn btn-primary btn-sm';
        btn.style.background = 'linear-gradient(90deg, #0f766e, #2563eb)';
        btn.textContent = '🔍 ' + tr('Analyze');
        btn.title = tr('Checks the dataset before training: duplicates, blurry or small images, faces that do not match, missing captions, captions that repeat the fixed traits of the person, and the shapes of the images.');
        btn.addEventListener('click', run);
        const forge = $('btn-ds-forge');
        if (forge) forge.after(btn); else bar.insertBefore(btn, bar.firstChild);
        const again = document.createElement('button');
        again.type = 'button';
        again.id = 'btn-ds-analysis-last';
        again.className = 'btn btn-secondary btn-sm';
        again.style.display = 'none';
        again.addEventListener('click', () => { const d = loadLast(); if (d) show(d); });
        btn.after(again);
        $('dataset_path')?.addEventListener('change', () => setTimeout(updateReopen, 0));
        updateReopen();

        // Al cerrar el editor de captions abierto desde el informe, se vuelve al informe.
        const modal = $('modal');
        if (modal) {
            new MutationObserver(() => {
                if (fromReport && !modal.classList.contains('active')) {
                    fromReport = false;
                    if (last && !last.error) show(last);
                }
            }).observe(modal, { attributes: true, attributeFilter: ['class'] });
        }

        box = document.createElement('div');
        box.className = 'da-box no-i18n';
        box.innerHTML = `<div class="da-win"><div class="da-head"><h3>🔍 ${esc(tr('Dataset analysis'))}</h3>
            <button type="button" data-da="again">${esc(tr('Analyze again'))}</button>
            <button type="button" data-da="close">${esc(tr('Close'))}</button></div><div class="da-body"></div>
            <div class="da-foot"><span class="da-sel"></span>
            <button type="button" class="da-apply" data-da="apply" disabled>${esc(tr('Apply changes'))}</button></div></div>`;
        document.body.appendChild(box);
        box.addEventListener('click', (e) => {
            const act = e.target.closest('[data-da]')?.dataset.da;
            if (act === 'close' || e.target === box) { box.style.display = 'none'; return; }
            if (act === 'again') return run();
            if (act === 'apply') return apply();
            if (act === 'character') return useCharacter();
            if (act === 'missing') return createMissing();
            const x = e.target.closest('.da-x');
            if (x) {
                // ✕ en la miniatura: marca o desmarca la imagen para quitarla del dataset.
                const f = x.dataset.file;
                if (sel.remove.has(f)) sel.remove.delete(f); else sel.remove.add(f);
                box.querySelectorAll(`.da-x[data-file="${CSS.escape(f)}"]`).forEach(b => b.classList.toggle('on', sel.remove.has(f)));
                return updateFoot();
            }
            const th = e.target.closest('.da-thumb');
            if (th) openImage(th.dataset.file);
        });
        box.addEventListener('change', (e) => {
            const c = e.target;
            if (c.dataset.opt === 'trigger') sel.trigger = c.checked;
            else if (c.dataset.phrase) { if (c.checked) sel.phrases.add(c.dataset.phrase); else sel.phrases.delete(c.dataset.phrase); }
            updateFoot();
        });
        document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && box.style.display === 'flex') box.style.display = 'none'; });
    }

    async function currentCaption(file) {
        // El caption de ahora (puede haber cambiado desde el análisis): de la lista del Dataset Manager o del servidor.
        const find = (list) => (list || []).find(i => i.file === file);
        let item = null;
        try { item = find(allDatasetImages); } catch (e) {}
        if (!item) {
            try { item = find((await (await fetch('/api/dataset-info?_=' + Date.now())).json()).images); } catch (e) {}
        }
        return item ? (item.caption || '') : ((last && last.captions_text && last.captions_text[file]) || '');
    }

    async function openImage(file) {
        if (typeof openModal !== 'function') return;
        box.style.display = 'none';
        const caption = await currentCaption(file);
        openModal('/api/dataset-image/' + encodeURIComponent(file) + '?v=' + Date.now(), file, caption);
        fromReport = true;
    }

    // ── Cambios propuestos: imágenes que salen del dataset, trigger que se añade y frases que se quitan ──
    let sel = { remove: new Set(), trigger: false, phrases: new Set() }, selFor = null;
    const CLASS_ONLY = /^(young |old |adult )?(woman|man|girl|boy|person|lady|guy)$/i;

    function suggest(data) {
        // Lo sugerido viene marcado: el segundo de cada par de duplicados, añadir el trigger, y quitar los rasgos
        // fijos repetidos (no la palabra de clase sola, como «young woman»).
        sel = { remove: new Set(), trigger: false, phrases: new Set() };
        for (const i of data.issues || []) {
            if (i.kind === 'duplicates') for (const [, b] of i.pairs || []) sel.remove.add(b);
            if (i.kind === 'no_trigger') sel.trigger = true;
            if (i.kind === 'repeated' && i.level === 'warn') {
                for (const [p] of i.removable || i.phrases || []) if (!CLASS_ONLY.test(p)) sel.phrases.add(p);
            }
        }
        selFor = data.when || data;
    }

    function updateFoot() {
        const parts = [];
        if (sel.remove.size) parts.push(tr('{n} images out of the dataset', { n: sel.remove.size }));
        if (sel.trigger) parts.push(tr('add the trigger'));
        if (sel.phrases.size) parts.push(tr('remove {n} phrases from the captions', { n: sel.phrases.size }));
        box.querySelector('.da-sel').textContent = parts.length ? parts.join(' · ') : tr('Nothing selected');
        box.querySelector('.da-apply').disabled = !parts.length;
    }

    async function apply() {
        const d = last;
        const trigger = ($('trigger_word')?.value || '').trim();
        const noTrig = (d.issues.find(i => i.kind === 'no_trigger') || {}).files || [];
        const lines = [];
        if (sel.remove.size) lines.push('• ' + tr('{n} images (and their captions) move to the _removed subfolder of the dataset: they leave the training, and you can bring them back.', { n: sel.remove.size }));
        if (sel.trigger && trigger) lines.push('• ' + tr('«{trigger}» is added at the start of {n} captions.', { trigger, n: noTrig.length }));
        if (sel.phrases.size) lines.push('• ' + tr('Removed from every caption: {phrases}.', { phrases: [...sel.phrases].map(p => '«' + p + '»').join(', ') }));
        const msg = tr('Apply these changes to the dataset?') + '\n\n' + lines.join('\n');
        const ok = typeof askConfirm === 'function' ? await askConfirm(msg) : confirm(msg);
        if (!ok) return;
        let res;
        try {
            res = await (await fetch('/api/dataset-fix', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ remove: [...sel.remove],
                                       add_trigger: sel.trigger && trigger ? { trigger, files: noTrig } : null,
                                       remove_phrases: [...sel.phrases] }) })).json();
        } catch (e) { res = { error: String(e) }; }
        if (res.error) { if (typeof toast === 'function') toast('Error: ' + res.error, 'error'); return; }
        if (typeof toast === 'function') {
            toast('✓ ' + tr('Done: {a} images out, trigger added to {b} captions, phrases removed from {c} captions.',
                            { a: res.removed.length, b: res.trigger, c: res.phrases }), 'success', 6000);
        }
        if (typeof loadDatasetInfo === 'function') loadDatasetInfo();
        run();
    }

    function useCharacter() {
        const s = $('caption_preset');
        if (!s) return;
        s.value = 'character';
        s.dispatchEvent(new Event('change', { bubbles: true }));
        box.style.display = 'none';
        $('caption_prompt_text')?.scrollIntoView({ block: 'center' });
        if (typeof toast === 'function') toast(tr('Character captions selected: press Redo Captions (with Overwrite) to write them again.'), 'info', 8000);
    }

    function createMissing() {
        const ow = $('caption_overwrite'), btn = $('btn-caption');
        if (!btn) return;
        if (ow) ow.checked = false;          // solo las imágenes sin caption
        box.style.display = 'none';
        btn.click();
    }

    function show(data) {
        last = data;
        box.style.display = 'flex';
        const body = box.querySelector('.da-body');
        box.querySelector('.da-foot').style.display = data.error ? 'none' : '';
        if (data.error) { body.innerHTML = `<div class="da-issue warn"><div class="da-msg">${esc(data.error)}</div></div>`; return; }
        if (selFor !== (data.when || data)) suggest(data);
        updateFoot();
        const when = data.when ? `<div class="da-when">${esc(tr('Analyzed {when}', { when: new Date(data.when).toLocaleString() }))}</div>` : '';
        body.innerHTML = when + summary(data) + (data.issues.length ? data.issues.map(issueHtml).join('')
            : `<div class="da-ok">✓ ${esc(tr('Nothing to fix: the dataset looks ready to train.'))}</div>`);
    }

    async function run() {
        const body = box.querySelector('.da-body');
        box.style.display = 'flex';
        body.innerHTML = `<div class="da-wait">⏳ ${esc(tr('Analyzing the dataset... With Face likeness installed, checking the faces takes a few seconds per 10 images.'))}</div>`;
        const trigger = ($('trigger_word')?.value || $('train_trigger_word')?.value || '').trim();
        let data;
        try {
            const res = await fetch('/api/dataset-analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                                              body: JSON.stringify({ trigger }) });
            data = await res.json();
        } catch (e) {
            data = { error: String(e) };
        }
        if (!data.error) {
            data.when = Date.now();
            saveLast(data);
            updateReopen();
        }
        show(data);
    }

    function summary(d) {
        const chips = [tr('{n} images', { n: d.count }), tr('{n} captions', { n: d.captions })];
        if (d.edit) chips.push(tr('edit pairs'));
        if (d.sizes) chips.push(tr('sides {a}–{b} px', { a: d.sizes.min, b: d.sizes.max }));
        for (const [shape, n] of d.shapes || []) chips.push(`${shape}: ${n}`);
        if (d.faces && d.faces.character) chips.push(tr('character: {n} faces, typical likeness {p}%', { n: d.faces.with_face, p: d.faces.median }));
        const warns = d.issues.filter(i => i.level === 'warn').length;
        chips.push(warns ? '⚠ ' + tr('{n} to check', { n: warns }) : '✓ ' + tr('no warnings'));
        return `<div class="da-sum">${chips.map(c => `<span class="da-chip">${esc(c)}</span>`).join('')}</div>`;
    }

    function issueHtml(i) {
        const files = i.files || [];
        const shown = files.slice(0, 30);
        const canRemove = !['no_caption', 'long', 'same_captions'].includes(i.kind);
        const thumbs = shown.map(f => {
            const score = i.scores && i.scores[f] != null ? `<span>${i.scores[f]}%</span>` : '';
            const x = canRemove ? `<button type="button" class="da-x${sel.remove.has(f) ? ' on' : ''}" data-file="${esc(f)}"
                title="${esc(tr('Take this image out of the dataset'))}">✕</button>` : '';
            return `<div class="da-thumb" data-file="${esc(f)}" title="${esc(f)}"><img loading="lazy" alt="" src="/api/dataset-image/${encodeURIComponent(f)}">${score}${x}</div>`;
        }).join('');
        const more = files.length > shown.length ? `<div class="da-more">+${files.length - shown.length}</div>` : '';
        const icon = i.level === 'warn' ? '⚠' : 'ℹ';
        const count = files.length ? ` <b>(${files.length})</b>` : '';
        // En los avisos que afectan a todo el dataset (frases repetidas) no se enseñan miniaturas.
        const showThumbs = i.kind !== 'repeated' && i.kind !== 'no_trigger';
        return `<div class="da-issue ${i.level}"><div class="da-msg">${icon} ${esc(i.msg)}${count}</div>
            ${showThumbs && files.length ? `<div class="da-thumbs">${thumbs}${more}</div>` : ''}${actions(i)}</div>`;
    }

    function actions(i) {
        const trigger = ($('trigger_word')?.value || '').trim();
        const rows = [];
        if (i.kind === 'no_trigger' && trigger) {
            rows.push(`<label><input type="checkbox" data-opt="trigger"${sel.trigger ? ' checked' : ''}>
                ${esc(tr('Add «{trigger}» at the start of these captions', { trigger }))}</label>`);
        }
        if (i.kind === 'repeated') {
            for (const [p] of i.removable || i.phrases || []) {
                rows.push(`<label><input type="checkbox" data-phrase="${esc(p)}"${sel.phrases.has(p) ? ' checked' : ''}>
                    ${esc(tr('Remove «{phrase}» from every caption', { phrase: p }))}</label>`);
            }
        }
        const buttons = [];
        if ((i.kind === 'repeated' && i.level === 'warn') || i.kind === 'long') {
            if ($('caption_preset')) buttons.push(`<button type="button" data-da="character">${esc(tr('Use Character captions'))}</button>`);
        }
        if (i.kind === 'no_caption' && $('btn-caption')) {
            buttons.push(`<button type="button" data-da="missing">${esc(tr('Create the missing captions'))}</button>`);
        }
        if (!rows.length && !buttons.length) return '';
        return `<div class="da-act">${rows.join('')}${buttons.length ? `<div class="da-btns">${buttons.join('')}</div>` : ''}</div>`;
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build);
    else build();
})();
