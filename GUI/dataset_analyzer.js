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
    let box = null, last = null;

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
    .da-wait { color: #94a3b8; font-size: .9rem; padding: 20px 4px; }`;

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

        box = document.createElement('div');
        box.className = 'da-box no-i18n';
        box.innerHTML = `<div class="da-win"><div class="da-head"><h3>🔍 ${esc(tr('Dataset analysis'))}</h3>
            <button type="button" data-da="again">${esc(tr('Analyze again'))}</button>
            <button type="button" data-da="close">${esc(tr('Close'))}</button></div><div class="da-body"></div></div>`;
        document.body.appendChild(box);
        box.addEventListener('click', (e) => {
            const act = e.target.closest('[data-da]')?.dataset.da;
            if (act === 'close' || e.target === box) box.style.display = 'none';
            else if (act === 'again') run();
            const th = e.target.closest('.da-thumb');
            if (th) openImage(th.dataset.file);
        });
        document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && box.style.display === 'flex') box.style.display = 'none'; });
    }

    function openImage(file) {
        if (typeof openModal !== 'function') return;
        box.style.display = 'none';
        const caption = (last && last.captions_text && last.captions_text[file]) || '';
        openModal('/api/dataset-image/' + encodeURIComponent(file) + '?v=' + Date.now(), file, caption);
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
        last = data;
        if (data.error) { body.innerHTML = `<div class="da-issue warn"><div class="da-msg">${esc(data.error)}</div></div>`; return; }
        body.innerHTML = summary(data) + (data.issues.length ? data.issues.map(issueHtml).join('')
            : `<div class="da-ok">✓ ${esc(tr('Nothing to fix: the dataset looks ready to train.'))}</div>`);
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
        const thumbs = shown.map(f => {
            const score = i.scores && i.scores[f] != null ? `<span>${i.scores[f]}%</span>` : '';
            return `<div class="da-thumb" data-file="${esc(f)}" title="${esc(f)}"><img loading="lazy" alt="" src="/api/dataset-image/${encodeURIComponent(f)}">${score}</div>`;
        }).join('');
        const more = files.length > shown.length ? `<div class="da-more">+${files.length - shown.length}</div>` : '';
        const icon = i.level === 'warn' ? '⚠' : 'ℹ';
        const count = files.length ? ` <b>(${files.length})</b>` : '';
        // En los avisos que afectan a todo el dataset (frases repetidas) no se enseñan miniaturas.
        const showThumbs = i.kind !== 'repeated' && i.kind !== 'no_trigger';
        return `<div class="da-issue ${i.level}"><div class="da-msg">${icon} ${esc(i.msg)}${count}</div>
            ${showThumbs && files.length ? `<div class="da-thumbs">${thumbs}${more}</div>` : ''}</div>`;
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build);
    else build();
})();
