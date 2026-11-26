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
    const IDS = { on: 'train_output_preservation', word: 'train_preservation_class',
                  weight: 'train_preservation_strength', every: 'train_preservation_every' };

    function update() {
        const hint = $('preservation_hint');
        if (!hint) return;
        const on = $(IDS.on).checked;
        [IDS.word, IDS.weight, IDS.every].forEach(id => { $(id).disabled = !on; });
        $('preservation_fields').style.display = on ? '' : 'none';
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
        }
        update();
    }

    if (typeof PreviewPrompts !== 'undefined') {
        PreviewPrompts.register({
            values: () => ($(IDS.on) ? {
                output_preservation: $(IDS.on).checked,
                preservation_class: $(IDS.word).value.trim(),
                preservation_strength: parseFloat($(IDS.weight).value) || 1.0,
                preservation_every: Math.max(1, parseInt($(IDS.every).value) || 1),
            } : {}),
            load: (data) => {
                if (!$(IDS.on)) return;
                $(IDS.on).checked = !!data.output_preservation;
                if (data.preservation_class !== undefined) $(IDS.word).value = data.preservation_class;
                if (data.preservation_strength !== undefined) $(IDS.weight).value = data.preservation_strength;
                if (data.preservation_every !== undefined) $(IDS.every).value = data.preservation_every;
                update();
            },
        });
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build);
    else build();
})();
