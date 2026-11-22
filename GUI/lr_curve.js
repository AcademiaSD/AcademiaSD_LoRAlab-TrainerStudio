// lr_curve.js — «New LR curve on resume» (issue 88) en todos los trainers.
// Casilla debajo de la fila del LR Scheduler. Marcada, al reanudar el LR scheduler empieza una curva nueva en
// ese paso: del Learning Rate actual al mínimo en el último paso, sin warmup. Ejemplo: 1000 pasos de 4e-4 a
// 4e-5 y, al subir Total Steps a 2000 y poner 1e-4, otros 1000 de 1e-4 a 1e-5. La lógica está en
// scripts/lora_options.py (LrCurve). Se guarda como lr_new_curve por el mismo camino que los Custom Prompts.
(function () {
    const ID = 'train_lr_new_curve';
    const $ = (id) => document.getElementById(id);

    function build() {
        const sched = $('train_lr_scheduler');
        if (!sched || $(ID)) return;
        const row = document.createElement('div');
        row.className = 'form-row';
        row.innerHTML = `
            <div class="form-group" title="When you Resume (for example after raising Total Steps), the LR curve starts again at that step: from the current Learning Rate down to its minimum at the last step, with no warmup. Example: 1000 steps from 4e-4 to 4e-5, then raise Total Steps to 2000, set the LR to 1e-4 and resume: the next 1000 steps go from 1e-4 to 1e-5. Off: the curve always covers the whole training, from step 0 to Total Steps, so resuming with more steps continues half-way along the longer curve. Resuming again with the same LR and Total Steps continues the same new curve. No effect with constant or on a new training.">
                <label style="display: flex; align-items: center; gap: 8px; margin: 0; cursor: pointer;" for="${ID}">
                    <input type="checkbox" id="${ID}" style="width: 18px; height: 18px; cursor: pointer;">
                    New LR curve on resume
                </label>
            </div>`;
        const after = sched.closest('.form-row') || sched.parentElement;
        after.after(row);
        if (typeof I18N !== 'undefined') I18N.apply(row);
        if (typeof LIVE_APPLY !== 'undefined' && LIVE_APPLY.resume) LIVE_APPLY.resume.lr_new_curve = ID;
    }

    if (typeof PreviewPrompts !== 'undefined') {
        PreviewPrompts.register({
            values: () => ($(ID) ? { lr_new_curve: $(ID).checked } : {}),
            load: (data) => { if ($(ID) && data.lr_new_curve !== undefined) $(ID).checked = !!data.lr_new_curve; },
        });
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build);
    else build();
})();
