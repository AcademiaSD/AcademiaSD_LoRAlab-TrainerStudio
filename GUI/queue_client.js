// queue_client.js — La cola de trabajos en cada entrenador (servidor: scripts/queue_link.py).
// The job queue in every trainer page.
//
// - Botón «Cola» junto a «Back»: cuántos trabajos esperan y un punto que late si hay uno en curso.
//   Lleva a la página de la cola del lanzador en la misma pestaña (sin ir abriendo pestañas nuevas;
//   «Back» vuelve). Sin lanzador no aparece.
// - Envuelve startScript(): con la GPU ocupada por otro trabajo, pre-caché y entrenamiento preguntan
//   «¿Añadir a la cola?» en vez de arrancar; los subtítulos avisan de que compartirán la GPU.
// - En un entrenador abierto por la cola para un trabajo, un aviso arriba lo dice: lo que se guarde
//   aquí se aplica solo a ese trabajo, que se pausa y continúa desde la cola.
// - Editor de un trabajo en espera (✎ en la cola): Start pasa a ser «Guardar cambios en la cola», que
//   guarda en la copia de ajustes del trabajo y vuelve a la cola; Cancelar lo deja como estaba.
(function () {
    const tr = (k, v) => (typeof t === 'function' ? t(k, v) : k.replace(/\{(\w+)\}/g, (m, n) => (v && n in v ? v[n] : m)));
    const KIND = { train: () => tr('training'), precache: () => tr('pre-cache'), caption: () => tr('captions') };
    let launcherUrl = '';
    let info = null;

    const CSS = `
    .qc-chip { position: relative; }
    .qc-chip .qc-count { min-width: 18px; height: 18px; padding: 0 5px; border-radius: 9px; background: #334155; color: #e2e8f0;
        font-size: 0.68rem; display: inline-flex; align-items: center; justify-content: center; font-variant-numeric: tabular-nums; }
    .qc-chip.qc-has .qc-count { background: linear-gradient(90deg, #0e7490, #6d28d9); color: #fff; }
    .qc-chip .qc-dot { width: 7px; height: 7px; border-radius: 50%; background: #a78bfa; box-shadow: 0 0 8px #a78bfa; display: none;
        animation: qc-pulse 1.6s ease-in-out infinite; }
    .qc-chip.qc-live .qc-dot { display: inline-block; }
    @keyframes qc-pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: .35; transform: scale(.7); } }
    .qc-banner { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin: 0 0 16px; padding: 12px 16px;
        border-radius: 10px; border: 1px solid #3b2f6e; color: #e2e8f0; font-size: 0.85rem;
        background: linear-gradient(90deg, rgba(14, 116, 144, 0.22), rgba(109, 40, 217, 0.22)); }
    .qc-banner b { color: #fff; }
    .qc-banner .qc-state { font-size: 0.72rem; padding: 2px 9px; border-radius: 999px; background: rgba(15, 23, 42, 0.6);
        border: 1px solid #334155; color: #c4b5fd; }
    .qc-banner .qc-note { color: #94a3b8; font-size: 0.78rem; flex: 1 1 260px; }
    .qc-banner a { margin-left: auto; color: #fff; text-decoration: none; font-weight: 700; font-size: 0.8rem; padding: 6px 12px;
        border-radius: 7px; background: linear-gradient(90deg, #0e7490, #6d28d9); white-space: nowrap; }
    .qc-newjob { margin-top: 10px; color: #fff !important; border: none; background: linear-gradient(90deg, #0e7490, #6d28d9) !important;
        box-shadow: 0 4px 16px rgba(109, 40, 217, 0.35); }
    .qc-newjob:hover { box-shadow: 0 0 0 1px #a78bfa, 0 6px 22px rgba(139, 92, 246, 0.55); }
    .qc-banner button { border: 1px solid #475569; background: rgba(15, 23, 42, 0.6); color: #e2e8f0; font-weight: 700;
        font-size: 0.8rem; padding: 6px 12px; border-radius: 7px; cursor: pointer; white-space: nowrap; }
    .qc-banner button.qc-save { border: none; color: #fff; background: linear-gradient(90deg, #0e7490, #6d28d9); }
    .qc-banner .qc-actions { margin-left: auto; display: flex; gap: 8px; }
    .qc-edit-save { color: #fff !important; border: none !important; background: linear-gradient(90deg, #0e7490, #6d28d9) !important; }
    `;

    function queueUrl() {
        return launcherUrl ? launcherUrl.replace(/\/$/, '') + '/queue' : '';
    }

    function openQueue() {
        const url = queueUrl();
        if (url) window.location.href = url;
    }

    async function post(url, body) {
        const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        let data = {};
        try { data = await res.json(); } catch (e) { /* respuesta vacía */ }
        return { ok: res.ok, data };
    }

    const STATE = {
        queued: () => tr('Waiting in the queue'), running: () => tr('Training now'), paused: () => tr('Paused'),
        done: () => tr('Done'), failed: () => tr('Failed'), stopped: () => tr('Stopped')
    };

    function render() {
        const chip = document.getElementById('qc-chip');
        if (chip) {
            chip.style.display = info && info.launcher && queueUrl() ? '' : 'none';
            if (info && info.launcher) {
                const n = (info.queued || 0) + (info.running ? 1 : 0);
                chip.querySelector('.qc-count').textContent = String(n);
                chip.classList.toggle('qc-has', n > 0);
                chip.classList.toggle('qc-live', !!info.running);
                chip.title = info.running
                    ? tr('Training now: {trainer} / {project}', { trainer: info.running.trainer_name, project: info.running.project || '-' })
                    : tr('Job queue');
            }
        }
        updateNewJob();
        keepTrainEnabled();
        if (info && info.edit_job) {
            renderEdit();
            return;
        }
        if (info && (info.job_id || info.draft)) {
            let banner = document.getElementById('qc-banner');
            if (!banner) {
                banner = document.createElement('div');
                banner.id = 'qc-banner';
                banner.className = 'qc-banner no-i18n';
                const header = document.querySelector('header');
                if (header) header.insertAdjacentElement('afterend', banner);
                else document.body.prepend(banner);
            }
            if (!info.job_id) {
                // Borrador: otro entrenamiento del mismo modelo, preparado mientras el primero entrena.
                banner.innerHTML = `<span>➕ <b>${escapeHtml(tr('New training for the queue'))}</b></span>`
                    + `<span class="qc-note">${escapeHtml(tr('What you save here applies only to this new job, not to the training in progress nor to your trainer settings. Press Start / Resume to add it to the queue.'))}</span>`
                    + (queueUrl() ? `<a href="${queueUrl()}">${escapeHtml(tr('Open the queue'))}</a>` : '');
                return;
            }
            const job = info.job || {};
            const state = STATE[job.state] ? STATE[job.state]() : '';
            banner.innerHTML = `<span>▶ ${tr('Queue job')}: <b>${escapeHtml(job.project || '-')}</b> · ${escapeHtml(KIND[job.kind] ? KIND[job.kind]() : '')}</span>`
                + (state ? `<span class="qc-state">${escapeHtml(state)}</span>` : '')
                + `<span class="qc-note">${escapeHtml(tr('What you save here applies only to this job, not to your trainer settings. Pause or resume it from the queue.'))}</span>`
                + (queueUrl() ? `<a href="${queueUrl()}">${escapeHtml(tr('Open the queue'))}</a>` : '');
        }
    }

    // ------------------------------------------------------------ edición
    function banner() {
        let el = document.getElementById('qc-banner');
        if (!el) {
            el = document.createElement('div');
            el.id = 'qc-banner';
            el.className = 'qc-banner no-i18n';
            const header = document.querySelector('header');
            if (header) header.insertAdjacentElement('afterend', el);
            else document.body.prepend(el);
        }
        return el;
    }

    function renderEdit() {
        const job = info.job || {};
        const el = banner();
        if (!el.dataset.lang || el.dataset.lang !== (window.I18N && I18N.lang) || el.dataset.project !== (job.project || '')) {
            el.dataset.lang = (window.I18N && I18N.lang) || '';
            el.dataset.project = job.project || '';
            el.innerHTML = `<span>✎ <b>${escapeHtml(tr('Editing a queued job'))}</b>: ${escapeHtml(job.project || '-')} · ${escapeHtml(KIND[job.kind] ? KIND[job.kind]() : '')}</span>`
                + `<span class="qc-note">${escapeHtml(tr('Changes are saved to this job only, not to your trainer settings. It does not start while you edit it.'))}</span>`
                + `<span class="qc-actions"><button type="button" class="qc-cancel">${escapeHtml(tr('Cancel'))}</button>`
                + `<button type="button" class="qc-save">💾 ${escapeHtml(tr('Save changes to the queue'))}</button></span>`;
            el.querySelector('.qc-cancel').addEventListener('click', () => finishEdit(false));
            el.querySelector('.qc-save').addEventListener('click', () => finishEdit(true));
        }
        // El botón de empezar del tipo de este trabajo guarda; el otro, Stop y «Nuevo» no hacen falta aquí.
        const mine = job.kind === 'precache' ? 'btn-precache' : 'btn-train';
        for (const id of ['btn-train', 'btn-precache']) {
            const btn = document.getElementById(id);
            if (!btn) continue;
            if (id !== mine) { show(btn, false); continue; }
            btn.classList.add('qc-edit-save');
            btn.style.gridColumn = '1 / -1';   // sola en su fila (Stop está oculto)
            const label = '💾 ' + tr('Save changes to the queue');
            const text = btn.querySelector('.main-text');
            if (text && text.textContent !== label) { text.textContent = label; text.classList.add('no-i18n'); }
        }
        const stop = document.getElementById('btn-stop');
        if (stop) show(stop, false);
    }

    async function finishEdit(save) {
        const buttons = document.querySelectorAll('#qc-banner button, #btn-train, #btn-precache');
        buttons.forEach(b => { b.disabled = true; });
        try {
            if (save) {
                // Start ya guardó su parte; se guardan las dos por si se cambió también la otra sección.
                if (typeof savePrecacheSettings === 'function') await savePrecacheSettings();
                if (typeof saveTrainSettings === 'function') await saveTrainSettings();
            }
            const res = await post('/api/queue-edit-done', { save });
            if (!res.ok || res.data.status !== 'ok') {
                toast(`[!] ${res.data.error || tr('Could not save it.')}`, 'error');
                return;
            }
            toast(save ? tr('Saved to the queue.') : tr('Changes discarded.'), 'success');
            setTimeout(openQueue, 400);
        } finally {
            buttons.forEach(b => { b.disabled = false; });
        }
    }

    // Mostrar / ocultar: los botones altos llevan display:flex !important en algunas páginas (pantallas
    // estrechas), así que se oculta también con !important.
    function show(el, on) {
        if (on) el.style.removeProperty('display');
        else el.style.setProperty('display', 'none', 'important');
    }

    function escapeHtml(s) {
        return String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    }

    // «Nuevo entrenamiento en cola»: mientras este entrena (Stop activo), abre un borrador del mismo
    // entrenador en otro puerto, con su propia copia de los ajustes, y navega a él en esta pestaña.
    function addNewJobButton() {
        const stop = document.getElementById('btn-stop');
        if (!stop || document.getElementById('qc-newjob')) return;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.id = 'qc-newjob';
        btn.className = 'btn btn-tall btn-full qc-newjob';
        show(btn, false);
        btn.title = tr('Opens this same trainer on another port with its own copy of the settings, to prepare another training and add it to the queue without touching the one in progress.');
        btn.innerHTML = `<span class="main-text">➕ <span>${escapeHtml(tr('New training for the queue'))}</span></span>`;
        btn.addEventListener('click', newJob);
        const row = stop.parentElement;
        row.insertAdjacentElement('afterend', btn);
        new MutationObserver(updateNewJob).observe(stop, { attributes: true, attributeFilter: ['disabled'] });
    }

    function updateNewJob() {
        const btn = document.getElementById('qc-newjob');
        const stop = document.getElementById('btn-stop');
        if (!btn || !stop) return;
        show(btn, !stop.disabled && info && info.launcher && !info.edit_job);
    }

    // Con la pre-caché en marcha en este entrenador, Start / Resume sigue disponible: el entrenamiento va a
    // la cola y empieza en cuanto acabe. (Entrenando, el siguiente se prepara con «Nuevo entrenamiento».)
    function keepTrainEnabled() {
        const btn = document.getElementById('btn-train');
        if (!btn) return;
        if (!btn.__qcObserved) {
            btn.__qcObserved = true;
            new MutationObserver(keepTrainEnabled).observe(btn, { attributes: true, attributeFilter: ['disabled'] });
        }
        if (btn.disabled && info && info.launcher && info.self_script === 'precache' && !info.edit_job) btn.disabled = false;
    }

    async function newJob() {
        const btn = document.getElementById('qc-newjob');
        btn.disabled = true;
        try {
            const res = await post('/api/queue-draft', {});
            if (!res.ok || res.data.status !== 'ok') {
                toast(`[!] ${res.data.error || tr('Could not open it.')}`, 'error');
                return;
            }
            const port = res.data.port;
            const url = `${location.protocol}//${location.hostname}:${port}`;
            const note = toast(tr('Opening a new {trainer} on port {port}...', { trainer: document.title.replace(/^AcademiaSD\s*/, '').replace(/\s*LoRAlab$/, ''), port }), 'info', 0);
            const end = Date.now() + 120000;
            while (Date.now() < end) {
                const r = await (await fetch(`/api/queue-draft-ready?port=${port}`, { cache: 'no-store' })).json().catch(() => ({}));
                if (r.ready) { window.location.href = url; return; }
                await new Promise(r2 => setTimeout(r2, 700));
            }
            note.remove();
            toast(`[!] ${tr('It did not start in time. Open it at {url}.', { url })}`, 'error');
        } finally {
            btn.disabled = false;
        }
    }

    async function refresh() {
        if (document.hidden) return;
        try { info = await (await fetch('/api/queue-info', { cache: 'no-store' })).json(); } catch (e) { info = null; }
        render();
    }

    function addChip() {
        const back = document.getElementById('back-to-launcher');
        if (!back || document.getElementById('qc-chip')) return;
        const chip = document.createElement('a');
        chip.id = 'qc-chip';
        chip.className = 'back-to-launcher qc-chip';
        chip.href = '#';
        chip.style.display = 'none';
        chip.style.marginLeft = '8px';
        chip.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" aria-hidden="true"><path d="M4 6h16M4 12h16M4 18h10"/></svg>`
            + `<span>${escapeHtml(tr('Queue'))}</span><span class="qc-dot"></span><span class="qc-count no-i18n">0</span>`;
        chip.addEventListener('click', e => { e.preventDefault(); openQueue(); });
        back.insertAdjacentElement('afterend', chip);
    }

    // Start / Start Pre-Cache / subtítulos: la cola decide si arranca ya o espera turno.
    function wrapStart() {
        const original = window.startScript;
        if (typeof original !== 'function' || original.__queue) return;
        const wrapped = async function (name, ...rest) {
            if (!['precache', 'train', 'caption'].includes(name)) return original(name, ...rest);
            let gate = null;
            try { gate = (await post('/api/queue-gate', { script: name })).data; } catch (e) { gate = null; }
            if (!gate || gate.mode === 'direct') return original(name, ...rest);

            if (gate.mode === 'edit') return finishEdit(true);
            if (gate.mode === 'queue_self') {
                const msg = gate.script === 'precache'
                    ? tr('The pre-cache is running in this trainer. Add this {kind} to the queue? It starts as soon as the pre-cache finishes.', { kind: KIND[name] ? KIND[name]() : name })
                    : tr('This trainer is busy. Add this {kind} to the queue? It starts as soon as the GPU is free.', { kind: KIND[name] ? KIND[name]() : name });
                if (await askConfirm(msg, { okLabel: tr('Add to queue'), cancelLabel: tr('Cancel') })) await addToQueue(name);
                return;
            }
            const r = gate.running || {};
            const who = { trainer: r.trainer_name || '', project: r.project || '-', kind: KIND[name] ? KIND[name]() : name };
            if (gate.mode === 'job') {
                if (await askConfirm(tr('This trainer runs a queue job: pause or resume it from the queue.'),
                                     { okLabel: tr('Open the queue'), cancelLabel: tr('Close') })) openQueue();
                return;
            }
            if (gate.mode === 'caption_busy') {
                const go = await askConfirm(tr('{trainer} is training ({project}). Creating captions now shares the GPU with it and can run out of VRAM. Continue?', who),
                                            { okLabel: tr('Create captions'), cancelLabel: tr('Cancel') });
                return go ? original(name, ...rest) : undefined;
            }
            // mode === 'queue'
            const add = await askConfirm(tr('{trainer} is training ({project}). Add this {kind} to the queue? It starts as soon as the GPU is free.', who),
                                         { okLabel: tr('Add to queue'), cancelLabel: tr('Cancel') });
            if (add) await addToQueue(name);
        };
        wrapped.__queue = true;
        window.startScript = wrapped;
    }

    async function addToQueue(name) {
        const res = await post('/api/queue-add', { kind: name });
        if (!res.ok || res.data.status !== 'ok') {
            toast(`[!] ${res.data.error || tr('Could not add it to the queue.')}`, 'error');
            return;
        }
        const pos = (res.data.pending || []).findIndex(j => j.id === res.data.job.id) + 1;
        toast(tr('Added to the queue: position {n}.', { n: pos || '?' }), 'success');
        if (typeof appendTerminal === 'function') {
            appendTerminal(`\n[SYSTEM] ${tr('Added to the queue: position {n}.', { n: pos || '?' })}`, false, 'term-success');
        }
        refresh();
    }

    function init() {
        const style = document.createElement('style');
        style.textContent = CSS;
        document.head.appendChild(style);
        addChip();
        addNewJobButton();
        wrapStart();
        fetch('/api/launcher-url').then(r => r.json()).then(d => { launcherUrl = d.url || ''; render(); }).catch(() => {});
        refresh();
        setInterval(refresh, 5000);
        document.addEventListener('visibilitychange', refresh);
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();
