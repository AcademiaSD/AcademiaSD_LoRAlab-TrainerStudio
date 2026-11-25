// xy_grid.js — Rejilla XY final (checkpoints × fuerza del LoRA), común a los trainers que la tienen.
// Final XY grid (checkpoints × LoRA strength), shared by the trainers that support it.
//
// Pone sus ajustes debajo de Preview Size (se guardan como xy_grid / xy_checkpoints / xy_strengths por el
// mismo camino que los Custom Prompts, y se aplican en caliente: el trainer los relee) y un visor encima de
// la galería con las rejillas del proyecto. La generación está en scripts/xy_grid.py.
(function () {
    const tr = (k, v) => (typeof t === 'function' ? t(k, v) : k.replace(/\{(\w+)\}/g, (m, n) => (v && n in v ? v[n] : m)));
    const $ = (id) => document.getElementById(id);
    const IDS = { on: 'train_xy_grid', rows: 'train_xy_checkpoints', cols: 'train_xy_strengths' };

    const CSS = `
    .xy-panel { margin: 0 0 14px; }
    .xy-panel .xy-list { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 10px; }
    .xy-panel a { display: block; border: 1px solid var(--panel-border, #334155); border-radius: 8px; overflow: hidden; background: #0b1222; }
    .xy-panel a:hover { border-color: var(--accent, #38bdf8); }
    .xy-panel img { display: block; width: 100%; height: 300px; object-fit: contain; }
    .xy-panel .xy-cap { padding: 6px 10px; font-size: 0.75rem; color: var(--text-muted, #94a3b8); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
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
            <div class="xy-hint" id="xy_grid_hint"></div>`;
        anchor.after(box);
        if (typeof I18N !== 'undefined') I18N.apply(box);
        [IDS.on, IDS.rows, IDS.cols].forEach(id => $(id).addEventListener('input', updateHint));
        $(IDS.on).addEventListener('change', updateHint);
        updateHint();
        if (typeof LIVE_APPLY !== 'undefined' && LIVE_APPLY.live) {
            LIVE_APPLY.live.xy_grid = IDS.on;
            LIVE_APPLY.live.xy_checkpoints = IDS.rows;
            LIVE_APPLY.live.xy_strengths = IDS.cols;
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
        panel.innerHTML = `<div class="panel-title" style="cursor:default;">${tr('Final XY grid')}</div><div class="xy-list"></div>`;
        gallery.before(panel);
        refresh();
        setInterval(refresh, 15000);
    }

    async function refresh() {
        const panel = $('xy_panel');
        if (!panel) return;
        let data;
        try { data = await (await fetch('/api/xy-grids', { cache: 'no-store' })).json(); } catch (e) { return; }
        const grids = (data && data.grids) || [];
        const key = grids.map(g => g.name + g.version).join('|');
        if (key === shown) return;
        shown = key;
        panel.style.display = grids.length ? '' : 'none';
        const list = panel.querySelector('.xy-list');
        list.innerHTML = '';
        for (const g of grids) {
            const url = `/api/xy-grid/${encodeURIComponent(g.name)}?v=${g.version}`;
            const a = document.createElement('a');
            a.href = url;
            a.target = '_blank';
            a.rel = 'noopener';
            a.title = g.prompt || g.name;
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
            } : {}),
            load: (data) => {
                if (!$(IDS.on)) return;
                if (data.xy_grid !== undefined) $(IDS.on).checked = !!data.xy_grid;
                if (data.xy_checkpoints) $(IDS.rows).value = data.xy_checkpoints;
                if (data.xy_strengths) $(IDS.cols).value = data.xy_strengths;
                updateHint();
            },
        });
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build);
    else build();
})();
