// optimizer.js — Selector de optimizador (AdamW, AdamW 8-bit, Prodigy, Adafactor), común a los trainers.
// Optimizer selector, shared by the trainers.
//
// Pone una fila «Optimizer» encima del LR Scheduler. Se guarda como optimizer por el mismo camino que los
// Custom Prompts y se aplica al reanudar (con otro optimizador, el estado guardado se descarta y el LoRA y el
// paso se conservan). Con Prodigy el campo Learning Rate no se usa: Prodigy calcula el LR. El optimizador por
// defecto del trainer va en el atributo data-default del <script> (AdamW si no lo lleva).
// La lógica está en scripts/lora_options.py (make_optimizer).
(function () {
    const ID = 'train_optimizer';
    const DEF = (document.currentScript && document.currentScript.dataset.default) || 'adamw';
    const tr = (k, v) => (typeof t === 'function' ? t(k, v) : k);
    const $ = (id) => document.getElementById(id);

    function update() {
        const sel = $(ID), lr = $('train_lr'), hint = $('train_optimizer_hint');
        if (!sel) return;
        const prodigy = sel.value === 'prodigy';
        // Con Prodigy el Learning Rate no se usa: el campo se oculta (su valor se guarda igual).
        const lrGroup = lr && (lr.closest('.form-group') || lr);
        if (lrGroup) lrGroup.style.display = prodigy ? 'none' : '';
        if (hint) {
            // Texto corto a la derecha del selector; la explicación completa, al pasar el ratón.
            const text = {
                prodigy: ['Finds the learning rate on its own.',
                          'Prodigy finds the learning rate: the Learning Rate field is not used. It starts very low and needs about 200-300 steps to find it, so use it with 1000 steps or more. The LR Scheduler still shapes it: with cosine, once found, it eases it down to the end. The console and the chart show the LR it finds. LoRA+ is not used with it.'],
                adamw8bit: ['Less VRAM.', 'Optimizer state in 8 bits: less VRAM, almost the same result.'],
                adafactor: ['Less VRAM than AdamW.', 'Less VRAM than AdamW. Use a similar Learning Rate.'],
            }[sel.value];
            hint.textContent = text ? tr(text[0]) : '';
            hint.title = text ? tr(text[1]) : '';
            hint.style.cursor = text ? 'help' : '';
        }
    }

    function build() {
        const sched = $('train_lr_scheduler');
        if (!sched || $(ID)) return;
        const row = document.createElement('div');
        row.className = 'form-row';
        row.innerHTML = `
            <div class="form-group" title="The algorithm that updates the LoRA weights. AdamW: the usual one. AdamW 8-bit: its state in 8 bits, less VRAM. Prodigy: finds the learning rate on its own, so you don't have to guess it. Adafactor: less VRAM than AdamW. Changing it on Resume starts a new optimizer state (the LoRA and the step are kept).">
                <label for="${ID}">Optimizer:</label>
                <select id="${ID}" class="no-i18n">
                    <option value="adamw">AdamW</option>
                    <option value="adamw8bit">AdamW 8-bit</option>
                    <option value="prodigy">Prodigy</option>
                    <option value="adafactor">Adafactor</option>
                </select>
            </div>
            <div class="form-group" style="display: flex; flex-direction: column; justify-content: flex-end;">
                <span id="train_optimizer_hint" style="font-size: 0.78rem; color: var(--text-muted, #94a3b8); line-height: 1.3; padding-bottom: 10px;"></span>
            </div>`;
        const before = sched.closest('.form-row') || sched.parentElement;
        before.before(row);
        $(ID).value = DEF;
        $(ID).addEventListener('change', update);
        if (typeof I18N !== 'undefined') I18N.apply(row);
        if (typeof LIVE_APPLY !== 'undefined' && LIVE_APPLY.resume) LIVE_APPLY.resume.optimizer = ID;
        update();
    }

    if (typeof PreviewPrompts !== 'undefined') {
        PreviewPrompts.register({
            values: () => ($(ID) ? { optimizer: $(ID).value } : {}),
            load: (data) => {
                if (!$(ID)) return;
                $(ID).value = data.optimizer || DEF;
                if (!$(ID).value) $(ID).value = DEF;
                update();
            },
        });
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build);
    else build();
})();
