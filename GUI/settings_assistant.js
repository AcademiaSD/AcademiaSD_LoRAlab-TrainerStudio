// settings_assistant.js — Asistente de ajustes («🪄 Suggest settings») en 2. Training Settings de los trainers de imagen.
// Settings assistant in the Training Settings of the image trainers.
//
// Pregunta qué se entrena (personaje, estilo, objeto) y propone pasos, learning rate, rank, warmup, cada cuánto
// guardar y hacer previews, y si conviene la máscara del sujeto, la protección de salida y los captions Character.
// Parte de los valores por defecto de ESE trainer (defaultValue de sus campos: el learning rate de Z-Image no es el
// de Anima) y del número de imágenes del dataset. Enseña el valor actual y el propuesto; se aplican los marcados.
(function () {
    const tr = (k, v) => (typeof t === 'function' ? t(k, v) : k.replace(/\{(\w+)\}/g, (m, n) => (v && n in v ? v[n] : m)));
    const $ = (id) => document.getElementById(id);
    const esc = (s) => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    let box = null, rows = [];

    // Pasos por imagen, mínimo y máximo (con acumulación 4), y multiplicadores de LR y rank por tipo.
    const TYPES = {
        character: { steps: 70, min: 600, max: 2500, lr: 1.0, rank: (r) => r, mask: true, captions: 'character' },
        style:     { steps: 40, min: 1000, max: 4000, lr: 0.75, rank: (r) => Math.min(32, Math.max(16, r * 2)), mask: false, captions: 'general' },
        object:    { steps: 60, min: 600, max: 2000, lr: 1.0, rank: (r) => r, mask: true, captions: 'general' },
    };

    const CSS = `
    .sa-box { position: fixed; inset: 0; background: rgba(2,6,23,.82); z-index: 2500; display: none; align-items: center; justify-content: center; padding: 20px; }
    .sa-win { background: var(--panel-bg, #0f172a); border: 1px solid var(--panel-border, #334155); border-radius: 12px; width: min(720px, 100%); max-height: 90vh; display: flex; flex-direction: column; }
    .sa-head { display: flex; align-items: center; gap: 10px; padding: 14px 18px; border-bottom: 1px solid var(--panel-border, #334155); }
    .sa-head h3 { margin: 0; font-size: 1.05rem; flex: 1; }
    .sa-body { overflow: auto; padding: 14px 18px; }
    .sa-q { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: flex-end; margin-bottom: 14px; }
    .sa-q label { display: flex; flex-direction: column; gap: 4px; font-size: .8rem; color: #94a3b8; }
    .sa-q select, .sa-q input { background: #0b1222; color: #e2e8f0; border: 1px solid #334155; border-radius: 6px; padding: 6px 8px; min-width: 120px; }
    .sa-types { display: flex; gap: 8px; }
    .sa-types button { padding: 8px 14px; border-radius: 8px; border: 1px solid #334155; background: #1e293b; color: #cbd5e1; cursor: pointer; }
    .sa-types button.on { background: linear-gradient(90deg, #7c3aed, #2563eb); color: #fff; border-color: transparent; }
    .sa-tab { width: 100%; border-collapse: collapse; font-size: .84rem; }
    .sa-tab th, .sa-tab td { padding: 6px 8px; border-bottom: 1px solid #1e293b; text-align: left; }
    .sa-tab th { color: #94a3b8; font-weight: 600; font-size: .76rem; }
    .sa-tab td.sa-new { color: #a7f3d0; font-weight: 600; }
    .sa-tab td.sa-same { color: #64748b; }
    .sa-tab .sa-why { display: block; color: #94a3b8; font-size: .72rem; font-weight: 400; }
    .sa-foot { display: flex; gap: 10px; justify-content: flex-end; padding: 12px 18px; border-top: 1px solid var(--panel-border, #334155); }
    .sa-foot button { padding: 8px 14px; border-radius: 8px; border: 1px solid #334155; background: #1e293b; color: #e2e8f0; cursor: pointer; }
    .sa-foot .sa-apply { border: none; background: linear-gradient(90deg, #059669, #2563eb); color: #fff; font-weight: 600; }`;

    function build() {
        const steps = $('train_total_steps');
        if (!steps || $('btn-settings-assistant')) return;
        const style = document.createElement('style');
        style.textContent = CSS;
        document.head.appendChild(style);
        const wrap = document.createElement('div');
        wrap.style.margin = '0 0 12px';
        wrap.innerHTML = `<button type="button" id="btn-settings-assistant" class="btn btn-primary btn-sm"
            style="background: linear-gradient(90deg, #7c3aed, #2563eb);">🪄 ${esc(tr('Suggest settings'))}</button>`;
        (steps.closest('.form-row') || steps.parentElement).before(wrap);
        $('btn-settings-assistant').title = tr('Proposes steps, learning rate, rank and the other settings for this model, for a character, a style or an object, from the number of images in the dataset.');
        $('btn-settings-assistant').addEventListener('click', open);

        box = document.createElement('div');
        box.className = 'sa-box no-i18n';
        box.innerHTML = `<div class="sa-win">
            <div class="sa-head"><h3>🪄 ${esc(tr('Suggest settings'))}</h3></div>
            <div class="sa-body">
              <div class="sa-q">
                <label>${esc(tr('What does the LoRA learn?'))}
                  <div class="sa-types">
                    <button type="button" data-type="character">👤 ${esc(tr('Character'))}</button>
                    <button type="button" data-type="style">🎨 ${esc(tr('Style'))}</button>
                    <button type="button" data-type="object">📦 ${esc(tr('Object'))}</button>
                  </div></label>
                <label>${esc(tr('Images'))}<input type="number" id="sa_images" min="1" max="10000"></label>
                <label class="sa-class">${esc(tr('The person is'))}
                  <select id="sa_class"><option value="woman">woman</option><option value="man">man</option><option value="person">person</option></select></label>
              </div>
              <table class="sa-tab"><thead><tr><th></th><th>${esc(tr('Setting'))}</th><th>${esc(tr('Now'))}</th><th>${esc(tr('Suggested'))}</th></tr></thead><tbody></tbody></table>
            </div>
            <div class="sa-foot"><button type="button" data-sa="close">${esc(tr('Cancel'))}</button>
              <button type="button" class="sa-apply" data-sa="apply">${esc(tr('Apply the ticked ones'))}</button></div></div>`;
        document.body.appendChild(box);
        box.addEventListener('click', (e) => {
            const tb = e.target.closest('[data-type]');
            if (tb) { box.querySelectorAll('[data-type]').forEach(b => b.classList.toggle('on', b === tb)); return render(); }
            const act = e.target.closest('[data-sa]')?.dataset.sa;
            if (act === 'close' || e.target === box) box.style.display = 'none';
            if (act === 'apply') apply();
        });
        box.addEventListener('input', (e) => { if (e.target.id === 'sa_images' || e.target.id === 'sa_class') render(); });
        box.addEventListener('change', (e) => { if (e.target.id === 'sa_class') render(); });
        document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && box.style.display === 'flex') box.style.display = 'none'; });
    }

    function open() {
        const n = parseInt($('ds-count')?.innerText) || 0;
        $('sa_images').value = n || 15;
        if (!box.querySelector('[data-type].on')) box.querySelector('[data-type="character"]').classList.add('on');
        const cls = ($('train_preservation_class')?.value || '').trim();
        if (['woman', 'man', 'person'].includes(cls)) $('sa_class').value = cls;
        box.style.display = 'flex';
        render();
    }

    const num = (id) => parseFloat($(id)?.value);
    const def = (id, fb) => { const v = parseFloat($(id)?.defaultValue); return isNaN(v) ? fb : v; };
    const round = (v, to) => Math.max(to, Math.round(v / to) * to);
    // Como se escribe en el campo (0.0004, 0.00002): tres cifras significativas, sin exponente salvo valores minúsculos.
    const fmtLr = (v) => String(Number(v.toPrecision(3)));

    function propose(type, images) {
        const T = TYPES[type];
        const accum = Math.max(1, num('train_grad_accum') || 4);
        let steps = Math.min(T.max, Math.max(T.min, images * T.steps));
        steps = round(steps * 4 / accum, 50);
        const baseLr = def('train_lr', 3e-4), baseRank = def('train_lora_rank', 16);
        const rank = T.rank(baseRank);
        const every = round(steps / 10, 50);
        const warm = Math.min(100, Math.max(20, round(steps * 0.05, 10)));
        const prodigy = $('train_optimizer')?.value === 'prodigy';
        const out = [];
        const add = (id, label, value, why, kind = 'value', checked = true) => {
            if (!$(id)) return;
            const el = $(id);
            const now = kind === 'check' ? el.checked : (el.tagName === 'SELECT' ? el.value : el.value);
            const a = parseFloat(now), b = parseFloat(value);
            const same = kind === 'check' ? now === value
                : (!isNaN(a) && !isNaN(b) && /^[\d.e+-]+$/i.test(String(now).trim()) ? Math.abs(a - b) <= Math.abs(b) * 1e-6
                   : String(now).trim() === String(value));
            out.push({ id, label, value, why, kind, now, same, checked: checked && !same });
        };
        add('train_total_steps', tr('Total steps'), steps,
            tr('{n} images × {k} steps each, with gradient accumulation {a}', { n: images, k: T.steps, a: accum }));
        if (prodigy) {
            add('train_optimizer', tr('Optimizer'), steps < 1000 ? 'adamw' : 'prodigy',
                steps < 1000 ? tr('Prodigy needs 1000 steps or more to find the learning rate') : tr('Prodigy finds the learning rate'));
        }
        if (!prodigy || steps < 1000) {
            add('train_lr', tr('Learning rate'), fmtLr(baseLr * T.lr),
                T.lr === 1 ? tr('This model\'s default') : tr('This model\'s default × {f}: styles overcook sooner', { f: T.lr }));
        }
        add('train_lora_rank', tr('LoRA rank'), rank, rank === baseRank ? tr('This model\'s default') : tr('A style needs more capacity'));
        add('train_lora_alpha', tr('LoRA alpha'), rank, tr('Same as the rank'));
        add('train_lr_scheduler', tr('LR scheduler'), 'cosine', tr('Eases the learning rate down to the end: cleaner last checkpoints'));
        add('train_warmup_steps', tr('Warmup steps'), warm, tr('About 5% of the steps'));
        add('train_save_every', tr('Save every'), every, tr('About 10 checkpoints to choose from'));
        add('train_preview_every', tr('Preview every'), every, tr('A preview with each checkpoint'));
        add('train_masked_loss', tr('Masked loss (subject)'), T.mask, T.mask
            ? tr('The LoRA learns the subject, not the backgrounds of the dataset')
            : tr('A style is the whole image: no mask'), 'check');
        if (type === 'character') {
            add('train_output_preservation', tr('Output preservation'), true,
                tr('Keeps «{w}» as the base model, so the LoRA does not leak into other people. About 2.2x slower: off unless you need it', { w: $('sa_class').value }),
                'check', false);
            add('train_preservation_class', tr('Class word'), $('sa_class').value, tr('For Output preservation'), 'value', false);
        }
        if ($('caption_preset')) {
            add('caption_preset', tr('Caption type'), T.captions, T.captions === 'character'
                ? tr('Short captions without the fixed traits: the trigger learns the identity. Then Redo Captions with Overwrite')
                : tr('Long descriptions for styles and objects'), 'value', type === 'character');
        }
        return out;
    }

    function render() {
        const type = box.querySelector('[data-type].on')?.dataset.type || 'character';
        box.querySelector('.sa-class').style.display = type === 'character' ? '' : 'none';
        const images = Math.max(1, parseInt($('sa_images').value) || 1);
        rows = propose(type, images);
        const show = (r, v) => r.kind === 'check' ? (v ? '✓' : '—') : v;
        box.querySelector('tbody').innerHTML = rows.map((r, i) => `<tr>
            <td><input type="checkbox" data-row="${i}"${r.checked ? ' checked' : ''}${r.same ? ' disabled' : ''}></td>
            <td>${esc(r.label)}<span class="sa-why">${esc(r.why)}</span></td>
            <td>${esc(show(r, r.now))}</td>
            <td class="${r.same ? 'sa-same' : 'sa-new'}">${esc(show(r, r.value))}</td></tr>`).join('');
    }

    function apply() {
        const ticked = [...box.querySelectorAll('[data-row]:checked')].map(c => rows[+c.dataset.row]);
        for (const r of ticked) {
            const el = $(r.id);
            if (r.kind === 'check') el.checked = !!r.value; else el.value = r.value;
            el.dispatchEvent(new Event('input', { bubbles: true }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
        }
        box.style.display = 'none';
        if (typeof toast === 'function') {
            toast('🪄 ' + tr('{n} settings applied. They are saved when you start the training.', { n: ticked.length }), 'success', 6000);
        }
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build);
    else build();
})();
