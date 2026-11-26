// xy_grid.js — Rejilla XY final (checkpoints × fuerza del LoRA) y mejor checkpoint, comunes a los trainers
// de imagen. Final XY grid (checkpoints × LoRA strength) and best checkpoint, shared by the image trainers.
//
// Pone sus ajustes debajo de Preview Size (se guardan como xy_grid / xy_checkpoints / xy_strengths /
// best_checkpoint / best_checkpoints por el mismo camino que los Custom Prompts, y se aplican en caliente:
// el trainer los relee) y un visor encima de la galería con el mejor checkpoint y las rejillas del proyecto.
// La generación está en scripts/xy_grid.py y scripts/best_checkpoint.py.
(function () {
    const tr = (k, v) => (typeof t === 'function' ? t(k, v) : k.replace(/\{(\w+)\}/g, (m, n) => (v && n in v ? v[n] : m)));
    const $ = (id) => document.getElementById(id);
    const IDS = { on: 'train_xy_grid', rows: 'train_xy_checkpoints', cols: 'train_xy_strengths',
                  best: 'train_best_checkpoint', bestN: 'train_best_checkpoints' };
    // Visor de imágenes dentro de la página (rejillas, hoja del mejor checkpoint, mapas de calor): una
    // pestaña nueva da problemas detrás de proxies como RunPod. Pantalla completa ajustada a la ventana;
    // clic en la imagen: tamaño real (con scroll); clic fuera, × o Esc: cerrar. Lo comparten xy_grid.js y
    // output_preservation.js (el primero que carga lo crea).
    window.LoRAlabViewer = window.LoRAlabViewer || (function () {
        let box = null;
        function close() { if (box) box.style.display = 'none'; }
        function build() {
            const style = document.createElement('style');
            style.textContent = `
            .lv-box { position: fixed; inset: 0; background: rgba(2,6,23,.9); z-index: 3000; display: none;
                      flex-direction: column; align-items: center; justify-content: center; padding: 18px; }
            .lv-scroll { flex: 1 1 auto; width: 100%; overflow: auto; display: flex; min-height: 0; }
            .lv-scroll img { margin: auto; max-width: 100%; max-height: 100%; object-fit: contain; cursor: zoom-in;
                             border-radius: 8px; box-shadow: 0 10px 40px rgba(0,0,0,.6); }
            .lv-box.lv-full .lv-scroll img { max-width: none; max-height: none; cursor: zoom-out; }
            .lv-cap { flex: 0 0 auto; margin-top: 10px; color: #cbd5e1; font-size: .85rem; max-width: 90%;
                      text-align: center; overflow-wrap: anywhere; }
            .lv-x { position: absolute; top: 10px; right: 16px; width: 38px; height: 38px; border-radius: 50%;
                    border: 1px solid #334155; background: #0f172a; color: #e2e8f0; font-size: 22px; cursor: pointer; }`;
            document.head.appendChild(style);
            box = document.createElement('div');
            box.className = 'lv-box no-i18n';
            box.innerHTML = `<button type="button" class="lv-x" aria-label="Close">×</button>
                <div class="lv-scroll"><img alt=""></div><div class="lv-cap"></div>`;
            document.body.appendChild(box);
            box.addEventListener('click', (e) => {
                if (e.target.tagName === 'IMG') box.classList.toggle('lv-full');
                else close();
            });
            document.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
        }
        return {
            open(src, caption) {
                if (!box) build();
                box.classList.remove('lv-full');
                box.querySelector('img').src = src;
                box.querySelector('.lv-cap').textContent = caption || '';
                box.style.display = 'flex';
            },
        };
    })();

    // Paneles de resultados encima de la galería (mapa de calor, rejilla XY): en una misma fila cuando hay
    // sitio, y cada uno plegable con su ▼ (se recuerda). Lo comparten xy_grid.js y output_preservation.js.
    window.LoRAlabPanels = window.LoRAlabPanels || {
        row(gallery) {
            let row = document.getElementById('result_panels');
            if (row) return row;
            const style = document.createElement('style');
            style.textContent = `
            #result_panels { display: flex; flex-wrap: wrap; gap: 8px 22px; align-items: flex-start; }
            #result_panels > * { min-width: 0; }
            #preservation_panel { flex: 0 1 470px; }
            #xy_panel { flex: 1 1 380px; }
            #result_panels .panel-title { cursor: pointer; justify-content: flex-start; gap: 8px; }
            .rp-caret { display: inline-block; font-size: .8em; transition: transform .15s; }
            .rp-min .rp-caret { transform: rotate(-90deg); }
            .rp-min { flex: 0 0 auto !important; }
            .rp-min > :not(.panel-title) { display: none !important; }`;
            document.head.appendChild(style);
            row = document.createElement('div');
            row.id = 'result_panels';
            // Encima de la barra de Face likeness, que va pegada a las previews.
            (document.getElementById('face-match-bar') || gallery).before(row);
            return row;
        },
        collapsible(panel, key) {
            const title = panel.querySelector('.panel-title');
            const caret = document.createElement('span');
            caret.className = 'rp-caret';
            caret.textContent = '▼';
            title.prepend(caret);
            const k = 'loralab_panel_' + key;
            try { if (localStorage.getItem(k) === '0') panel.classList.add('rp-min'); } catch (e) {}
            title.addEventListener('click', () => {
                panel.classList.toggle('rp-min');
                try { localStorage.setItem(k, panel.classList.contains('rp-min') ? '0' : '1'); } catch (e) {}
            });
        },
    };

    const esc = (s) => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

    const CSS = `
    .xy-panel { margin: 0 0 14px; }
    .xy-panel .xy-list { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 10px; }
    .xy-panel a.xy-item { display: block; border: 1px solid var(--panel-border, #334155); border-radius: 8px; overflow: hidden; background: #0b1222; }
    .xy-panel a.xy-item:hover { border-color: var(--accent, #38bdf8); }
    .xy-panel img { display: block; width: 100%; height: 300px; object-fit: contain; }
    .xy-panel .xy-cap { padding: 6px 10px; font-size: 0.75rem; color: var(--text-muted, #94a3b8); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .xy-best { overflow-x: auto; margin: 0 0 12px; padding: 10px 12px; background: #0b1222; border: 1px solid var(--panel-border, #334155); border-radius: 8px; font-size: 0.82rem; color: #cbd5e1; }
    .xy-best-head { display: flex; align-items: center; gap: 8px 12px; flex-wrap: wrap; margin-bottom: 8px; }
    .xy-best-head b { color: #fbbf24; font-size: 0.95rem; }
    .xy-best-head .xy-file { color: var(--text-muted, #94a3b8); flex: 1 1 200px; }
    .xy-best button { padding: 7px 12px; border-radius: 6px; border: none; font-weight: 600; cursor: pointer; font-size: 0.82rem;
        background: linear-gradient(90deg, #d97706, #f59e0b); color: #fff; min-height: 34px; }
    .xy-best button:disabled { opacity: .4; cursor: not-allowed; }
    .xy-best table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
    .xy-best th, .xy-best td { padding: 3px 6px; text-align: right; border-bottom: 1px solid #1e293b; white-space: nowrap; }
    .xy-best th { color: var(--text-muted, #94a3b8); font-weight: 600; font-size: 0.74rem; }
    .xy-best th:first-child, .xy-best td:first-child { text-align: left; }
    .xy-best tr.xy-star td { color: #fbbf24; font-weight: 700; }
    .xy-best td .xy-bar { display: inline-block; height: 8px; border-radius: 4px; background: linear-gradient(90deg, #22d3ee, #a78bfa); vertical-align: middle; margin-right: 6px; }
    .xy-hint { font-size: 0.72rem; color: var(--text-muted, #94a3b8); margin-top: -2px; line-height: 1.35; }`;

    function strengths() {
        return ($(IDS.cols)?.value || '').split(/[,;]/).map(s => parseFloat(s)).filter(v => !isNaN(v) && v >= 0 && v <= 3);
    }

    function updateHint() {
        const hint = $('xy_grid_hint');
        if (!hint) return;
        const on = $(IDS.on).checked;
        $(IDS.rows).disabled = $(IDS.cols).disabled = !on;
        const rows = Math.max(1, Math.min(8, parseInt($(IDS.rows).value) || 4));
        const cols = Math.max(1, strengths().length);
        hint.textContent = on ? tr('{n} images per prompt at the end of the training (Custom Prompts, or the first caption).', { n: rows * cols }) : '';
        const best = $(IDS.best).checked;
        $(IDS.bestN).disabled = !best;
        const n = Math.max(2, Math.min(16, parseInt($(IDS.bestN).value) || 8));
        const save = parseInt($('train_save_every')?.value) || 0;
        $('xy_best_hint').textContent = !best ? '' : (save <= 0 ? tr('Needs Save Every: it chooses among the saved checkpoints.')
            : tr('{n} images at the end of the training: 4 per checkpoint and 4 without the LoRA. SigLIP (~0.8 GB) is downloaded the first time.', { n: (n + 1) * 4 }));
    }

    function build() {
        const anchor = $('preview_size_hint') || $('train_preview_size')?.closest('.form-row');
        if (!anchor || $(IDS.on)) return;
        const style = document.createElement('style');
        style.textContent = CSS;
        document.head.appendChild(style);

        const box = document.createElement('div');
        box.innerHTML = `
            <div class="form-row" style="margin-top:10px;">
                <div class="form-group" title="When the training ends, the same image (same prompt and seed) is made with several checkpoints (rows) and LoRA strengths (columns), and joined into one image per prompt, shown above the previews and saved in the xy_grid folder of the output. It shows which step and strength to use without going to ComfyUI. The strength works as in ComfyUI. It adds time at the end: one preview per image. With Face likeness installed, each image shows its % and the console says which one is the most alike.">
                    <label style="display: flex; align-items: center; gap: 8px; margin: 0; cursor: pointer;" for="${IDS.on}">
                        <input type="checkbox" id="${IDS.on}" style="width: 18px; height: 18px; cursor: pointer;">
                        Final XY grid
                    </label>
                </div>
            </div>
            <div class="form-row">
                <div class="form-group" title="Rows: checkpoints spread over the training (with 4 and 800 steps: 200, 400, 600 and the final one). They are saved while training, so turn it on before the training ends.">
                    <label>Checkpoints:</label>
                    <input type="number" id="${IDS.rows}" value="4" min="1" max="8">
                </div>
                <div class="form-group" title="Columns: LoRA strengths, separated by commas (0 to 3). 0 shows the model without the LoRA.">
                    <label>Strengths:</label>
                    <input type="text" id="${IDS.cols}" value="0.75, 1.0, 1.25">
                </div>
            </div>
            <div class="xy-hint" id="xy_grid_hint"></div>
            <div class="form-row" style="margin-top:10px;">
                <div class="form-group" title="When the training ends, 4 fixed images (2 prompts × 2 seeds, or 1 prompt × 4 seeds) are made with the model without the LoRA and with several saved checkpoints, and each checkpoint is scored: likeness (face likeness for characters, SigLIP similarity to the dataset for styles and objects), prompt adherence (SigLIP) and variety between seeds. A checkpoint that loses the prompt or makes every seed the same is overcooked. The recommended one is copied as _BEST_LoRA.safetensors and shown above the previews, with the scores of every checkpoint. It is a heuristic: check the image sheet.">
                    <label style="display: flex; align-items: center; gap: 8px; margin: 0; cursor: pointer;" for="${IDS.best}">
                        <input type="checkbox" id="${IDS.best}" style="width: 18px; height: 18px; cursor: pointer;">
                        Find the best checkpoint
                    </label>
                </div>
                <div class="form-group" title="Saved checkpoints to test, spread over the training and taken from Save Every (the final one is always included). More = finer choice, and 4 more images for each one.">
                    <label>Checkpoints to test:</label>
                    <input type="number" id="${IDS.bestN}" value="8" min="2" max="16">
                </div>
            </div>
            <div class="xy-hint" id="xy_best_hint"></div>`;
        anchor.after(box);
        if (typeof I18N !== 'undefined') I18N.apply(box);
        Object.values(IDS).forEach(id => { $(id).addEventListener('input', updateHint); $(id).addEventListener('change', updateHint); });
        $('train_save_every')?.addEventListener('input', updateHint);
        updateHint();
        if (typeof LIVE_APPLY !== 'undefined' && LIVE_APPLY.live) {
            LIVE_APPLY.live.xy_grid = IDS.on;
            LIVE_APPLY.live.xy_checkpoints = IDS.rows;
            LIVE_APPLY.live.xy_strengths = IDS.cols;
            LIVE_APPLY.live.best_checkpoint = IDS.best;
            LIVE_APPLY.live.best_checkpoints = IDS.bestN;
        }
        buildViewer();
    }

    // ── Visor encima de la galería ──
    let shown = '';
    function buildViewer() {
        const gallery = $('gallery');
        if (!gallery || $('xy_panel')) return;
        const panel = document.createElement('div');
        panel.className = 'xy-panel no-i18n';
        panel.id = 'xy_panel';
        panel.style.display = 'none';
        panel.innerHTML = `<div class="panel-title">${esc(tr('Final XY grid'))}</div>
            <div class="xy-best" id="xy_best" style="display:none;"></div><div class="xy-list"></div>`;
        window.LoRAlabPanels.row(gallery).append(panel);
        window.LoRAlabPanels.collapsible(panel, 'xy');
        refresh();
        setInterval(refresh, 15000);
    }

    function bestHtml(b) {
        const face = b.mode === 'face';
        const rows = [Object.assign({}, b.base, { step: null, score: null })].concat(b.checkpoints || []);
        const scores = (b.checkpoints || []).map(r => r.score);
        const lo = Math.min(0, ...scores), hi = Math.max(...scores, 1e-6);
        const body = rows.map(r => {
            const star = r.step === b.best;
            const bar = r.score == null ? '' : `<span class="xy-bar" style="width:${Math.round(40 * (r.score - lo) / (hi - lo || 1))}px"></span>`;
            return `<tr class="${star ? 'xy-star' : ''}"><td>${r.step == null ? esc(tr('No LoRA')) : (star ? '★ ' : '') + r.step}</td>
                <td>${r.resemblance.toFixed(1)}${face ? '%' : ''}</td><td>${r.adherence.toFixed(2)}</td>
                <td>${r.seed_sim.toFixed(1)}</td><td>${bar}${r.score == null ? '' : r.score.toFixed(2)}</td></tr>`;
        }).join('');
        const file = b.exists ? tr('Copied as {file}', { file: b.file }) : tr('Its LoRA file was not found.');
        return `<div class="xy-best-head"><b>${esc(tr('Best checkpoint: step {n}', { n: b.best }))}</b>
                <span class="xy-file">${esc(file)}</span>
                <button type="button" id="xy_best_send" ${b.exists ? '' : 'disabled'}>${esc(tr('Send best to Models'))}</button></div>
            <table><thead><tr><th>${esc(tr('Step'))}</th><th>${esc(face ? tr('Face likeness') : tr('Similarity to the dataset'))}</th>
                <th>${esc(tr('Prompt adherence'))}</th><th>${esc(tr('Seed similarity'))}</th><th>${esc(tr('Score'))}</th></tr></thead>
            <tbody>${body}</tbody></table>`;
    }

    async function sendBest() {
        const btn = $('xy_best_send');
        btn.disabled = true;
        try {
            const res = await fetch('/api/xy-best/export', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ target_dir: ($('export_models_dir')?.value || '').trim(),
                                       final_name: ($('export_final_name')?.value || '').trim() }),
            });
            const data = await res.json();
            if (data.status === 'ok') toast(`✓ ${tr('Best checkpoint (step {n}) sent to: {path}', { n: data.step, path: data.dest })}`, 'success');
            else toast(`[!] ${data.error || res.status}`, 'error');
        } catch (e) {
            toast(`[!] ${e}`, 'error');
        } finally {
            btn.disabled = false;
        }
    }

    async function refresh() {
        const panel = $('xy_panel');
        if (!panel) return;
        let data, best;
        try {
            [data, best] = await Promise.all([fetch('/api/xy-grids', { cache: 'no-store' }).then(r => r.json()),
                                              fetch('/api/xy-best', { cache: 'no-store' }).then(r => r.json())]);
        } catch (e) { return; }
        const grids = (data && data.grids) || [];
        const hasBest = best && best.best != null;
        const key = grids.map(g => g.name + g.version).join('|') + (hasBest ? `#${best.created}${best.exists}` : '');
        if (key === shown) return;
        shown = key;
        panel.style.display = grids.length || hasBest ? '' : 'none';
        const bestBox = $('xy_best');
        bestBox.style.display = hasBest ? '' : 'none';
        if (hasBest) {
            bestBox.innerHTML = bestHtml(best);
            $('xy_best_send').addEventListener('click', sendBest);
        }
        const list = panel.querySelector('.xy-list');
        list.innerHTML = '';
        for (const g of grids) {
            const url = `/api/xy-grid/${encodeURIComponent(g.name)}?v=${g.version}`;
            const a = document.createElement('a');
            a.className = 'xy-item';
            a.href = url;
            a.title = g.prompt || g.name;
            // Se abre en el visor de la página, no en una pestaña nueva.
            a.addEventListener('click', (e) => {
                e.preventDefault();
                window.LoRAlabViewer.open(url, `${g.name}${g.prompt ? ' — ' + g.prompt : ''}`);
            });
            a.innerHTML = `<img loading="lazy" alt=""><div class="xy-cap"></div>`;
            a.querySelector('img').src = url;
            a.querySelector('.xy-cap').textContent = `${g.name}${g.prompt ? ' — ' + g.prompt : ''}`;
            list.appendChild(a);
        }
    }

    if (typeof PreviewPrompts !== 'undefined') {
        PreviewPrompts.register({
            values: () => ($(IDS.on) ? {
                xy_grid: $(IDS.on).checked,
                xy_checkpoints: Math.max(1, Math.min(8, parseInt($(IDS.rows).value) || 4)),
                xy_strengths: $(IDS.cols).value.trim() || '0.75, 1.0, 1.25',
                best_checkpoint: $(IDS.best).checked,
                best_checkpoints: Math.max(2, Math.min(16, parseInt($(IDS.bestN).value) || 8)),
            } : {}),
            load: (data) => {
                if (!$(IDS.on)) return;
                if (data.xy_grid !== undefined) $(IDS.on).checked = !!data.xy_grid;
                if (data.xy_checkpoints) $(IDS.rows).value = data.xy_checkpoints;
                if (data.xy_strengths) $(IDS.cols).value = data.xy_strengths;
                if (data.best_checkpoint !== undefined) $(IDS.best).checked = !!data.best_checkpoint;
                if (data.best_checkpoints) $(IDS.bestN).value = data.best_checkpoints;
                updateHint();
            },
        });
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build);
    else build();
})();
