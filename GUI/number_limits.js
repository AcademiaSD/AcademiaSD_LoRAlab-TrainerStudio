// number_limits.js — Los campos numéricos de los trainers no aceptan valores sin sentido.
// Number fields in the trainers do not accept meaningless values.
//
// Pone min (y step="any" donde hacen falta decimales) a los campos que no lo traen ya en su página, y
// al cambiar cualquier campo numérico (al salir de él o con Intro) corrige su valor: por debajo del
// mínimo, el mínimo; por encima del máximo, el máximo; vacío o no numérico, su valor inicial; con step
// entero, redondeado. Los scripts aplican los mismos límites al cargar el json (scripts/settings_limits.py).
(function () {
    // id del campo: [mínimo, decimales permitidos]
    const LIMITS = {
        precache_max_side:      [64, false],
        precache_target_area:   [4096, false],
        precache_max_seq_len:   [1, false],
        train_total_steps:      [1, false],
        train_batch_size:       [1, false],
        train_grad_accum:       [1, false],
        train_warmup_steps:     [0, false],
        train_lr_cycles:        [1, false],
        train_lora_rank:        [1, false],
        train_lora_alpha:       [1, true],
        train_lokr_factor:      [1, false],
        train_save_every:       [0, false],
        train_preview_every:    [0, false],
        train_preview_steps:    [1, false],
        train_preview_cfg:      [0, true],
        train_turbo_strength:   [0, true],
        train_guidance:         [0, true],
        train_preview_cfg_high: [0, true],
        train_vram_budget_gb:   [0, true],
        train_vram_headroom_gb: [0, true],
        train_vram_swap_gb:     [0, true],
    };

    function applyAttributes() {
        for (const [id, [min, decimals]] of Object.entries(LIMITS)) {
            const el = document.getElementById(id);
            if (!el || el.type !== 'number') continue;
            // Lo que ya traiga la página para ese modelo manda (p. ej. LoKr Factor con mínimo 2).
            if (!el.hasAttribute('min')) el.min = String(min);
            if (decimals && !el.hasAttribute('step')) el.step = 'any';
        }
    }

    function fix(el) {
        const text = el.value.trim();
        let value = text === '' ? NaN : Number(text);
        if (!Number.isFinite(value)) {
            // Campo vacío o con algo que no es un número: vuelve a su valor inicial si lo tiene.
            if (el.defaultValue !== '') el.value = el.defaultValue;
            return;
        }
        const min = el.min === '' ? -Infinity : Number(el.min);
        const max = el.max === '' ? Infinity : Number(el.max);
        let fixed = Math.min(max, Math.max(min, value));
        const step = el.step === '' ? 1 : Number(el.step);   // sin step, el navegador usa 1
        if (el.step !== 'any' && Number.isInteger(step)) fixed = Math.round(fixed);
        if (fixed !== value) el.value = String(fixed);
    }

    // Fase de captura: se corrige antes que los onchange de la propia página lean el valor.
    document.addEventListener('change', (e) => {
        const el = e.target;
        if (el instanceof HTMLInputElement && el.type === 'number') fix(el);
    }, true);

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', applyAttributes);
    else applyAttributes();
})();
