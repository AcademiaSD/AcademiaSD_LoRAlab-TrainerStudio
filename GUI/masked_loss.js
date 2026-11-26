// masked_loss.js — Pérdida con máscara del sujeto (BiRefNet), común a los trainers.
// Masked loss (subject mask from BiRefNet), shared by the trainers.
//
// Pone su casilla y el peso del fondo debajo de la protección de salida (o del LR Scheduler) y, con la
// casilla marcada, un panel encima de las previews con la hoja de máscaras (masked_loss_preview.png). Se
// guardan como masked_loss / mask_background por el mismo camino que los Custom Prompts y se aplican al
// empezar o reanudar. La lógica está en scripts/subject_masks.py.
(function () {
    const tr = (k, v) => (typeof t === 'function' ? t(k, v) : k.replace(/\{(\w+)\}/g, (m, n) => (v && n in v ? v[n] : m)));
    const $ = (id) => document.getElementById(id);
    const IDS = { on: 'train_masked_loss', bg: 'train_mask_background' };

    function update() {
        const on = $(IDS.on)?.checked;
        if (!$(IDS.bg)) return;
        $(IDS.bg).disabled = !on;
        $('masked_loss_fields').style.display = on ? '' : 'none';
        showSheet();
    }

    function build() {
        const sched = $('train_lr_scheduler');
        if (!sched || $(IDS.on)) return;
        const anchor = $('preservation_hint') || $('train_lr_new_curve')?.closest('.form-row') || sched.closest('.form-row');
        const box = document.createElement('div');
        box.innerHTML = `
            <div class="form-row">
                <div class="form-group" title="The LoRA learns the subject, not the backgrounds of the dataset. Before the training starts, BiRefNet (MIT, about 0.2 GB, downloaded the first time) cuts out the subject of every image (person, object...), and the loss weighs 1 on the subject and the Background weight elsewhere. Useful with small datasets or with backgrounds that repeat. The masks are made once (only the images that change are made again) and a sheet with all of them shows above the previews.">
                    <label style="display: flex; align-items: center; gap: 8px; margin: 0; cursor: pointer;" for="${IDS.on}">
                        <input type="checkbox" id="${IDS.on}" style="width: 18px; height: 18px; cursor: pointer;">
                        Masked loss (subject)
                    </label>
                </div>
            </div>
            <div class="form-row" id="masked_loss_fields" style="display: none;">
                <div class="form-group" title="How much the background still teaches, from 0 to 1. 0.1 by default: the subject leads. 0 can leave halos or odd backgrounds; higher learns more of the setting (light, mood).">
                    <label for="${IDS.bg}">Background weight:</label>
                    <input type="number" id="${IDS.bg}" value="0.1" min="0" max="1" step="0.05">
                </div>
                <div class="form-group" style="display: flex; flex-direction: column; justify-content: flex-end;">
                    <span style="font-size: 0.78rem; color: var(--text-muted, #94a3b8); line-height: 1.3; padding-bottom: 10px;">BiRefNet cuts out the subject before training.</span>
                </div>
            </div>`;
        anchor.after(...box.children);
        if (typeof I18N !== 'undefined') ['masked_loss_fields', IDS.on].forEach(id => I18N.apply($(id).closest('.form-row')));
        $(IDS.on).addEventListener('change', update);
        if (typeof LIVE_APPLY !== 'undefined' && LIVE_APPLY.resume) {
            LIVE_APPLY.resume.masked_loss = IDS.on;
            LIVE_APPLY.resume.mask_background = IDS.bg;
        }
        buildPanel();
        update();
    }

    // ── Hoja de máscaras encima de las previews (en la fila de paneles de resultados) ──
    let shown = '';
    function buildPanel() {
        const gallery = $('gallery');
        if (!gallery || $('mask_panel') || !window.LoRAlabPanels) return;
        const panel = document.createElement('div');
        panel.id = 'mask_panel';
        panel.style.display = 'none';
        panel.style.flex = '0 1 420px';
        panel.innerHTML = `<div class="panel-title">Subject masks</div>
            <a href="#"><img alt="" style="display:block; max-width:100%; max-height:360px; border-radius:10px; border:1px solid var(--panel-border, #334155);"></a>
            <div style="font-size:0.78rem; color:var(--text-muted, #94a3b8); margin-top:6px; line-height:1.35;">Bright: the subject the LoRA learns. Dark blue: background (weighs the Background weight).</div>`;
        window.LoRAlabPanels.row(gallery).append(panel);
        window.LoRAlabPanels.collapsible(panel, 'masks');
        if (typeof I18N !== 'undefined') I18N.apply(panel);
        panel.querySelector('a').addEventListener('click', (e) => {
            e.preventDefault();
            const src = panel.querySelector('img').src;
            if (src && window.LoRAlabViewer) window.LoRAlabViewer.open(src, tr('Subject masks'));
        });
        setInterval(showSheet, 15000);
    }

    async function showSheet() {
        const panel = $('mask_panel');
        if (!panel) return;
        if (!$(IDS.on)?.checked) { panel.style.display = 'none'; shown = ''; return; }
        let info;
        try { info = await fetch('/api/masked-loss-preview', { cache: 'no-store' }).then(r => r.json()); } catch (e) { return; }
        panel.style.display = info.exists ? '' : 'none';
        const key = String(info.version);
        if (!info.exists || key === shown) return;
        shown = key;
        panel.querySelector('img').src = `/api/masked-loss-preview.png?v=${info.version}`;
    }

    if (typeof PreviewPrompts !== 'undefined') {
        PreviewPrompts.register({
            values: () => ($(IDS.on) ? {
                masked_loss: $(IDS.on).checked,
                mask_background: (v => (isNaN(v) ? 0.1 : Math.max(0, Math.min(1, v))))(parseFloat($(IDS.bg).value)),
            } : {}),
            load: (data) => {
                if (!$(IDS.on)) return;
                $(IDS.on).checked = !!data.masked_loss;
                if (data.mask_background !== undefined) $(IDS.bg).value = data.mask_background;
                update();
            },
        });
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build);
    else build();
})();
