// output_preservation.js — Output preservation (protege lo que el modelo ya sabe), común a los trainers.
// Output preservation (keeps what the model already knows), shared by the trainers.
//
// Pone sus ajustes debajo de la fila del LR Scheduler (y de «New LR curve on resume»): casilla, palabra de
// clase, fuerza y cada cuántos pasos, con un aviso del coste. Se guardan como output_preservation /
// preservation_class / preservation_strength / preservation_every por el mismo camino que los Custom Prompts
// y se aplican al empezar o reanudar. La lógica está en scripts/output_preservation.py.
(function () {
    const tr = (k, v) => (typeof t === 'function' ? t(k, v) : k.replace(/\{(\w+)\}/g, (m, n) => (v && n in v ? v[n] : m)));
    const $ = (id) => document.getElementById(id);
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

    const IDS = { on: 'train_output_preservation', word: 'train_preservation_class',
                  weight: 'train_preservation_strength', every: 'train_preservation_every',
                  map: 'train_preservation_map' };

    function update() {
        const hint = $('preservation_hint');
        if (!hint) return;
        const on = $(IDS.on).checked;
        [IDS.word, IDS.weight, IDS.every, IDS.map].forEach(id => { $(id).disabled = !on; });
        $('preservation_fields').style.display = on ? '' : 'none';
        showMaps();
        if (!on) { hint.textContent = ''; return; }
        if (!$(IDS.word).value.trim()) {
            hint.style.color = '#fbbf24';
            hint.textContent = tr('Write the class word (woman, man, dog, style...): without it, it is not used.');
            return;
        }
        hint.style.color = '';
        const n = Math.max(1, Math.min(100, parseInt($(IDS.every).value) || 1));
        hint.textContent = tr('About {x}x slower: one more pass without the LoRA and one with it every {n} step(s). The class prompts are encoded before the training starts.',
            { x: (1 + 1.25 / n).toFixed(1), n });
    }

    function build() {
        const sched = $('train_lr_scheduler');
        if (!sched || $(IDS.on)) return;
        const anchor = $('train_lr_new_curve')?.closest('.form-row') || sched.closest('.form-row') || sched.parentElement;
        const box = document.createElement('div');
        box.innerHTML = `
            <div class="form-row">
                <div class="form-group" title="Keeps what the model already knows. Every step, the caption is also used with the trigger replaced by a class word (for example 'woman'), and the LoRA is penalised if it changes the model's output for it. The LoRA learns the trigger, but 'woman' stays the base model's woman: it does not leak into other people, it overcooks less and it mixes better with other LoRAs. Like the Differential Output Preservation of ai-toolkit. It makes the training slower.">
                    <label style="display: flex; align-items: center; gap: 8px; margin: 0; cursor: pointer;" for="${IDS.on}">
                        <input type="checkbox" id="${IDS.on}" style="width: 18px; height: 18px; cursor: pointer;">
                        Output preservation
                    </label>
                </div>
            </div>
            <div class="form-row" id="preservation_fields" style="display: none;">
                <div class="form-group" title="The generic word for what the trigger is: woman, man, person, dog, car, style... It replaces the trigger in the captions.">
                    <label for="${IDS.word}">Class word:</label>
                    <input type="text" id="${IDS.word}" class="no-i18n" placeholder="woman">
                </div>
                <div class="form-group" title="How strongly the base model's output is kept. 1.0 by default; higher keeps more and learns the trigger a bit slower.">
                    <label for="${IDS.weight}">Strength:</label>
                    <input type="number" id="${IDS.weight}" value="1.0" min="0" max="10" step="0.1">
                </div>
                <div class="form-group" title="Every how many steps it is applied. 1 is every step (about 2x slower); 2, every other step (about 1.5x).">
                    <label for="${IDS.every}">Every N steps:</label>
                    <input type="number" id="${IDS.every}" value="1" min="1" max="100">
                </div>
                <div class="form-group" style="display: flex; flex-direction: column; justify-content: flex-end; padding-bottom: 10px;" title="Every 100 steps, saves a heatmap over a dataset image: where the LoRA changes the base model's output when the prompt has the class word instead of the trigger. Usually the face and the hair; if the clothes, the background or the style light up too, the LoRA is learning things from the dataset it should not. Shown above the previews.">
                    <label style="display: flex; align-items: center; gap: 8px; margin: 0; cursor: pointer;" for="${IDS.map}">
                        <input type="checkbox" id="${IDS.map}" style="width: 18px; height: 18px; cursor: pointer;">
                        Preservation heatmap
                    </label>
                </div>
            </div>
            <div id="preservation_hint" style="font-size: 0.78rem; color: var(--text-muted, #94a3b8); line-height: 1.35; margin: -4px 0 10px;"></div>`;
        anchor.after(...box.children);
        Object.values(IDS).forEach(id => $(id).addEventListener('input', update));
        $(IDS.on).addEventListener('change', update);
        if (typeof I18N !== 'undefined') ['preservation_fields', IDS.on].forEach(id => I18N.apply($(id).closest('.form-row')));
        if (typeof LIVE_APPLY !== 'undefined' && LIVE_APPLY.resume) {
            LIVE_APPLY.resume.output_preservation = IDS.on;
            LIVE_APPLY.resume.preservation_class = IDS.word;
            LIVE_APPLY.resume.preservation_strength = IDS.weight;
            LIVE_APPLY.resume.preservation_every = IDS.every;
            LIVE_APPLY.resume.preservation_map = IDS.map;
        }
        buildViewer();
        update();
    }

    // ── Panel de mapas de calor encima de la galería (solo con la casilla marcada) ──
    let shown = '';
    function buildViewer() {
        const gallery = $('gallery');
        if (!gallery || $('preservation_panel')) return;
        const style = document.createElement('style');
        style.textContent = `
        .pres-panel { margin: 0 0 14px; }
        .pres-panel .pres-main { display: flex; gap: 12px; align-items: flex-start; flex-wrap: wrap; }
        .pres-panel .pres-main img { max-height: 420px; max-width: 100%; border-radius: 10px; border: 1px solid var(--panel-border, #334155); }
        .pres-panel .pres-thumbs { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }
        .pres-panel .pres-thumbs img { height: 70px; border-radius: 6px; cursor: pointer; opacity: .65; border: 1px solid transparent; }
        .pres-panel .pres-thumbs img.on, .pres-panel .pres-thumbs img:hover { opacity: 1; border-color: var(--accent, #38bdf8); }
        .pres-panel .pres-note { font-size: 0.78rem; color: var(--text-muted, #94a3b8); max-width: 260px; line-height: 1.4; }`;
        document.head.appendChild(style);
        const panel = document.createElement('div');
        panel.className = 'pres-panel';
        panel.id = 'preservation_panel';
        panel.style.display = 'none';
        panel.innerHTML = `<div class="panel-title" style="cursor:default;">Preservation heatmap</div>
            <div class="pres-main"><a href="#"><img alt=""></a>
            <div class="pres-note">Where the LoRA changes the base model when the prompt has the class word instead of the trigger. Bright: it changes a lot; dark: it keeps the base model.</div></div>
            <div class="pres-thumbs"></div>`;
        (($('xy_panel') && $('xy_panel').parentElement === gallery.parentElement) ? $('xy_panel') : gallery).before(panel);
        if (typeof I18N !== 'undefined') I18N.apply(panel);
        panel.querySelector('.pres-main a').addEventListener('click', (e) => {
            e.preventDefault();
            const a = e.currentTarget;
            if (a.dataset.src) window.LoRAlabViewer.open(a.dataset.src, a.dataset.caption);
        });
        setInterval(showMaps, 15000);
    }

    function select(item) {
        const panel = $('preservation_panel');
        const url = `/api/preservation-map/${encodeURIComponent(item.name)}?v=${item.version}`;
        const link = panel.querySelector('.pres-main a');
        link.href = url;
        link.dataset.src = url;
        link.dataset.caption = tr('Step {n}', { n: item.step });
        panel.querySelector('.pres-main img').src = url;
        panel.querySelectorAll('.pres-thumbs img').forEach(im => im.classList.toggle('on', im.dataset.name === item.name));
    }

    async function showMaps() {
        const panel = $('preservation_panel');
        if (!panel) return;
        if (!($(IDS.on)?.checked && $(IDS.map)?.checked)) { panel.style.display = 'none'; shown = ''; return; }
        let data;
        try { data = await fetch('/api/preservation-maps', { cache: 'no-store' }).then(r => r.json()); } catch (e) { return; }
        const maps = (data && data.maps) || [];
        panel.style.display = maps.length ? '' : 'none';
        const key = maps.map(m => m.name + m.version).join('|');
        if (!maps.length || key === shown) return;
        shown = key;
        const thumbs = panel.querySelector('.pres-thumbs');
        thumbs.innerHTML = '';
        for (const m of maps) {
            const im = document.createElement('img');
            im.loading = 'lazy';
            im.dataset.name = m.name;
            im.title = tr('Step {n}', { n: m.step });
            im.src = `/api/preservation-map/${encodeURIComponent(m.name)}?v=${m.version}`;
            im.addEventListener('click', () => select(m));
            thumbs.appendChild(im);
        }
        select(maps[0]);
    }

    if (typeof PreviewPrompts !== 'undefined') {
        PreviewPrompts.register({
            values: () => ($(IDS.on) ? {
                output_preservation: $(IDS.on).checked,
                preservation_class: $(IDS.word).value.trim(),
                preservation_strength: parseFloat($(IDS.weight).value) || 1.0,
                preservation_every: Math.max(1, parseInt($(IDS.every).value) || 1),
                preservation_map: $(IDS.map).checked,
            } : {}),
            load: (data) => {
                if (!$(IDS.on)) return;
                $(IDS.on).checked = !!data.output_preservation;
                if (data.preservation_class !== undefined) $(IDS.word).value = data.preservation_class;
                if (data.preservation_strength !== undefined) $(IDS.weight).value = data.preservation_strength;
                if (data.preservation_every !== undefined) $(IDS.every).value = data.preservation_every;
                $(IDS.map).checked = !!data.preservation_map;
                update();
            },
        });
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build);
    else build();
})();
