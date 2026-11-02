// face_match.js — Parecido de cara (%) de cada preview con el dataset, común a los entrenadores de imagen.
// Face likeness (%) of each preview against the dataset, shared by the image trainers.
//
// Se coloca sola encima de la galería (#gallery). El modelo (InsightFace, solo uso no comercial) NO se
// descarga por defecto: el botón abre una ventana con dos casillas (bajo mi responsabilidad / sin fines
// comerciales) y «Descargar» solo se activa con las dos marcadas. Instalado, pone una etiqueta con el %
// en cada preview; el cálculo lo hace el servidor (scripts/face_match.py) en CPU.
// window.FaceMatch (active / get / want) deja a la gráfica (loss_chart.js) poner el % sobre sus previews.
// Sin hover ni nada de escritorio: funciona igual en el móvil.
(function () {
    const tr = (k, v) => (typeof t === 'function' ? t(k, v) : k.replace(/\{(\w+)\}/g, (m, n) => (v && n in v ? v[n] : m)));

    function store(key, value) {
        try {
            if (value === undefined) return JSON.parse(localStorage.getItem('faceMatch.' + key));
            localStorage.setItem('faceMatch.' + key, JSON.stringify(value));
        } catch (e) { return null; }
    }

    const CSS = `
    .fm-bar { display: flex; align-items: center; gap: 8px 12px; flex-wrap: wrap; margin: 0 0 10px; padding: 8px 12px;
        background: #0b1222; border: 1px solid var(--panel-border, #334155); border-radius: 8px; font-size: 0.82rem; color: #cbd5e1; }
    .fm-bar b { color: #e2e8f0; }
    .fm-msg { color: var(--text-muted, #94a3b8); flex: 1 1 200px; min-width: 0; }
    .fm-bar label { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; }
    .fm-bar input[type=checkbox] { width: 18px; height: 18px; }
    .fm-btn { padding: 7px 12px; border-radius: 6px; border: none; font-weight: 600; cursor: pointer; font-size: 0.82rem;
        background: linear-gradient(90deg, #0e7490, #6d28d9); color: #fff; min-height: 34px; }
    .fm-btn.fm-sec { background: #334155; }
    .fm-btn:disabled { opacity: .4; cursor: not-allowed; }
    .fm-progress { flex: 1 1 160px; height: 8px; background: #1e293b; border-radius: 4px; overflow: hidden; }
    .fm-progress i { display: block; height: 100%; width: 0; background: linear-gradient(90deg, #22d3ee, #a78bfa); transition: width .4s; }
    .fm-error { color: #f87171; }
    .fm-badge { position: absolute; top: 6px; left: 6px; z-index: 2; padding: 3px 8px; border-radius: 999px; font-size: 0.78rem;
        font-weight: 700; color: #fff; background: rgba(15, 23, 42, 0.85); border: 1px solid #475569; pointer-events: none; }
    .fm-badge.fm-hi { background: rgba(5, 150, 105, 0.9); border-color: #34d399; }
    .fm-badge.fm-mid { background: rgba(180, 83, 9, 0.9); border-color: #fbbf24; }
    .fm-badge.fm-lo { background: rgba(185, 28, 28, 0.9); border-color: #f87171; }
    .fm-badge.fm-none { color: #94a3b8; }
    .fm-modal { position: fixed; inset: 0; z-index: 3000; display: none; align-items: center; justify-content: center;
        padding: 16px; background: rgba(0, 0, 0, 0.8); overflow-y: auto; }
    .fm-modal.fm-open { display: flex; }
    .fm-dialog { width: 100%; max-width: 540px; max-height: calc(100vh - 32px); overflow-y: auto; box-sizing: border-box;
        background: #0f172a; border: 1px solid #334155; border-radius: 12px; padding: 20px; color: #e2e8f0; font-size: 0.9rem; line-height: 1.5; }
    .fm-dialog h3 { margin: 0 0 10px; font-size: 1.05rem; }
    .fm-dialog p { margin: 0 0 10px; color: #cbd5e1; }
    .fm-dialog .fm-warn { background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 8px; padding: 10px 12px; color: #fde68a; }
    .fm-check { display: flex; align-items: flex-start; gap: 10px; padding: 10px 12px; margin: 8px 0; border: 1px solid #334155;
        border-radius: 8px; cursor: pointer; background: #111a2e; }
    .fm-check input { width: 22px; height: 22px; flex: 0 0 auto; margin: 0; cursor: pointer; }
    .fm-actions { display: flex; gap: 10px; margin-top: 14px; flex-wrap: wrap; }
    .fm-actions .fm-btn { flex: 1 1 160px; min-height: 42px; font-size: 0.9rem; }
    .fm-dialog a { color: #38bdf8; word-break: break-all; }
    `;

    let bar, modal, gallery;
    let st = null;            // último /api/face-match/status
    let statusTimer = null;
    const cache = {};         // src (con ?v=) -> porcentaje | null
    const byName = {};        // archivo -> porcentaje | null, para la gráfica (loss_chart.js)
    let busy = false, again = false;
    let reference = null;

    async function api(url, body) {
        const res = await fetch(url, body === undefined ? { cache: 'no-store' }
            : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        const data = await res.json().catch(() => ({}));
        if (!res.ok && !data.error) data.error = 'HTTP ' + res.status;
        return data;
    }

    function enabled() { return store('show') !== false; }
    function active() { return !!(st && st.installed && enabled()); }
    // La gráfica se redibuja cuando llegan porcentajes o cambia «Mostrar %».
    function notify() { document.dispatchEvent(new Event('facematch-update')); }

    // Pide al servidor los que falten, de 60 en 60 (su límite por petición).
    async function fetchScores(names) {
        const out = {};
        for (let i = 0; i < names.length; i += 60) {
            const data = await api('/api/face-match/scores', { files: names.slice(i, i + 60) });
            if (!data.ok) throw new Error(data.error || 'Error');
            reference = data.reference;
            Object.assign(out, data.scores);
        }
        names.forEach(n => { byName[n] = n in out ? out[n] : null; });
        notify();
        return out;
    }

    // API para loss_chart.js: get(archivo) = % | null (sin cara) | undefined (aún no se sabe).
    const wanted = new Set();
    let chartBusy = false;
    async function want(names) {
        if (!active()) return;
        names.filter(n => /\.(png|jpe?g|webp|bmp)$/i.test(n) && !(n in byName)).forEach(n => wanted.add(n));
        if (chartBusy || !wanted.size) return;
        chartBusy = true;
        const todo = [...wanted];
        wanted.clear();
        try { await fetchScores(todo); } catch (e) { todo.forEach(n => { byName[n] = null; }); }
        chartBusy = false;
        if (wanted.size) want([]);
    }
    window.FaceMatch = { active, get: name => byName[name], want };

    function cards() {
        return [...gallery.querySelectorAll('.preview-card')].map(card => {
            const img = card.querySelector('img');
            return img ? { card, img, name: img.getAttribute('alt') || '', src: img.getAttribute('src') || '' } : null;
        }).filter(c => c && c.name);
    }

    function paint() {
        const show = st && st.installed && enabled();
        for (const c of cards()) {
            let badge = c.card.querySelector('.fm-badge');
            if (!show || !(c.src in cache)) { if (badge) badge.remove(); continue; }
            if (!badge) {
                badge = document.createElement('span');
                badge.className = 'fm-badge no-i18n';
                c.card.appendChild(badge);
            }
            const v = cache[c.src];
            badge.className = 'fm-badge no-i18n ' + (v == null ? 'fm-none' : v >= 60 ? 'fm-hi' : v >= 40 ? 'fm-mid' : 'fm-lo');
            badge.textContent = v == null ? '👤 ' + tr('no face') : '👤 ' + Math.round(v) + '%';
        }
    }

    async function refreshScores() {
        if (!st || !st.installed || !enabled()) { paint(); return; }
        if (busy) { again = true; return; }
        const todo = cards().filter(c => !(c.src in cache));
        if (!todo.length) { paint(); return; }
        busy = true;
        renderBar(tr('Analyzing faces...'));
        try {
            const res = await fetchScores(todo.map(c => c.name));
            // Lo que el servidor no devuelve (vídeo, archivo que ya no está) no se vuelve a pedir.
            todo.forEach(c => { cache[c.src] = c.name in res ? res[c.name] : null; });
            renderBar();
        } catch (e) { renderBar(null, e.message || String(e)); }
        busy = false;
        paint();
        if (again) { again = false; refreshScores(); }
    }

    function renderBar(working, error) {
        if (!st) return;
        const title = `<b>👤 ${tr('Face likeness')}</b>`;
        if (st.downloading) {
            const label = st.stage === 'onnxruntime' ? tr('Installing onnxruntime...')
                : tr('Downloading the face model... {pct}%', { pct: Math.round((st.progress || 0) * 100) });
            bar.innerHTML = `${title}<span class="fm-msg">${label}</span>
                <span class="fm-progress"><i style="width:${st.stage === 'onnxruntime' ? 5 : Math.round((st.progress || 0) * 100)}%"></i></span>`;
            return;
        }
        if (!st.installed) {
            bar.innerHTML = `${title}<span class="fm-msg">${st.error ? `<span class="fm-error">${escapeHtml(st.error)}</span>`
                : tr('Compares the face in each preview with the dataset. Optional: not downloaded by default.')}</span>
                <button type="button" class="fm-btn" data-fm="open">${tr('Enable...')}</button>`;
            return;
        }
        let msg;
        if (error) msg = `<span class="fm-error">${escapeHtml(error)}</span>`;
        else if (working) msg = working;
        else if (reference && !reference.faces) msg = tr('No faces found in the dataset images.');
        else if (reference) msg = tr('Reference: {faces} faces in {images} dataset images. The same person usually scores above 40-50%.', reference);
        else msg = tr('Similarity with the faces of the dataset.');
        bar.innerHTML = `${title}<label><input type="checkbox" data-fm="show" ${enabled() ? 'checked' : ''}> ${tr('Show %')}</label>
            <span class="fm-msg">${msg}</span>`;
    }

    function escapeHtml(s) {
        return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    }

    async function loadStatus() {
        try { st = await api('/api/face-match/status'); } catch (e) { return; }
        renderBar();
        clearTimeout(statusTimer);
        if (st.downloading) statusTimer = setTimeout(loadStatus, 1500);
        else if (st.installed) { refreshScores(); notify(); }
    }

    function openModal() {
        modal.querySelectorAll('input[type=checkbox]').forEach(cb => { cb.checked = false; });
        syncModal();
        modal.classList.add('fm-open');
    }
    function closeModal() { modal.classList.remove('fm-open'); }
    function syncModal() {
        const all = [...modal.querySelectorAll('input[type=checkbox]')].every(cb => cb.checked);
        modal.querySelector('[data-fm=download]').disabled = !all;
    }

    async function download() {
        const btn = modal.querySelector('[data-fm=download]');
        btn.disabled = true;
        const data = await api('/api/face-match/download', {
            own_responsibility: modal.querySelector('[data-fm=resp]').checked,
            non_commercial: modal.querySelector('[data-fm=noncom]').checked,
        }).catch(e => ({ error: String(e) }));
        if (!data.ok) { syncModal(); alert(data.error || 'Error'); return; }
        closeModal();
        loadStatus();
    }

    function build() {
        gallery = document.getElementById('gallery');
        if (!gallery || document.getElementById('face-match-bar')) return;
        const style = document.createElement('style');
        style.textContent = CSS;
        document.head.appendChild(style);

        bar = document.createElement('div');
        bar.id = 'face-match-bar';
        bar.className = 'fm-bar no-i18n';
        gallery.insertAdjacentElement('beforebegin', bar);
        bar.addEventListener('click', e => {
            if (e.target.closest('[data-fm=open]')) openModal();
        });
        bar.addEventListener('change', e => {
            if (e.target.matches('[data-fm=show]')) { store('show', e.target.checked); refreshScores(); paint(); notify(); }
        });

        modal = document.createElement('div');
        modal.className = 'fm-modal no-i18n';
        modal.innerHTML = `
            <div class="fm-dialog" role="dialog" aria-modal="true">
                <h3>👤 ${tr('Face likeness: terms of use')}</h3>
                <p>${tr('The comparison uses the InsightFace models (SCRFD detector and ArcFace recognizer, about 280 MB) and the onnxruntime package. They run on the CPU of the machine with the trainer, also on RunPod.')}</p>
                <p class="fm-warn">${tr('The InsightFace pretrained models are licensed for non-commercial research purposes only. Their authors, not this program, set those terms.')}
                    <br><a href="https://github.com/deepinsight/insightface#license" target="_blank" rel="noopener">github.com/deepinsight/insightface#license</a></p>
                <label class="fm-check"><input type="checkbox" data-fm="resp"><span>${tr('I accept that I download and use these models under my own responsibility.')}</span></label>
                <label class="fm-check"><input type="checkbox" data-fm="noncom"><span>${tr('I will not use them for commercial purposes.')}</span></label>
                <div class="fm-actions">
                    <button type="button" class="fm-btn fm-sec" data-fm="cancel">${tr('Cancel')}</button>
                    <button type="button" class="fm-btn" data-fm="download" disabled>⬇ ${tr('Download')}</button>
                </div>
            </div>`;
        document.body.appendChild(modal);
        modal.addEventListener('change', syncModal);
        modal.addEventListener('click', e => {
            if (e.target === modal || e.target.closest('[data-fm=cancel]')) closeModal();
            else if (e.target.closest('[data-fm=download]')) download();
        });
        document.addEventListener('keydown', e => { if (e.key === 'Escape' && modal.classList.contains('fm-open')) closeModal(); });

        // La galería se redibuja entera cuando llega una preview: se vuelven a poner las etiquetas.
        new MutationObserver(() => refreshScores()).observe(gallery, { childList: true });
        loadStatus();
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build);
    else build();
})();
