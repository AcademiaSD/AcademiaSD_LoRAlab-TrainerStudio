// preview_prompts.js — Hasta 4 prompts propios para las previews (Custom Prompts).
// Up to 4 custom prompts for the previews.
//
// Cada página de trainer tiene el selector Caption Mode (train_preview_caption_mode) y el bloque
// custom_prompt_container con la caja del prompt 1 (train_preview_custom_prompt). Este script:
//   - quita «Custom prompt» del Caption Mode y añade «Custom Prompts» (0-4) y «One random prompt»;
//   - con 1-4 muestra ese número de cajas (las 2-4 se crean aquí) y el Caption Mode deja de usarse;
//   - PreviewPrompts.values() da las claves nuevas para el json y PreviewPrompts.load(datos) las pone;
//   - añade las claves a LIVE_APPLY (aviso de Save JSON con el entrenamiento en marcha).
// Las claves del json y lo que hace cada trainer están en scripts/preview_prompts.py.
const PreviewPrompts = (function () {
    const MAX = 4;
    const tr = (k, v) => (typeof t === 'function' ? t(k, v) : k);
    const $ = (id) => document.getElementById(id);
    const promptId = (i) => (i === 1 ? 'train_preview_custom_prompt' : `train_preview_custom_prompt_${i}`);
    const promptKey = (i) => (i === 1 ? 'preview_custom_prompt' : `preview_custom_prompt_${i}`);
    let ready = false;

    function count() {
        const el = $('train_preview_custom_count');
        const n = el ? parseInt(el.value, 10) : 0;
        return Number.isFinite(n) ? Math.max(0, Math.min(MAX, n)) : 0;
    }

    // Muestra las cajas de los prompts en uso y deja el Caption Mode solo cuando sirve.
    function refresh() {
        if (!ready) return;
        const n = count();
        const container = $('custom_prompt_container');
        if (container) container.style.display = n > 0 ? 'block' : 'none';
        for (let i = 1; i <= MAX; i++) {
            const box = $(promptId(i));
            const group = box && (i === 1 ? box.closest('.pp-prompt') || box.parentElement : box.closest('.pp-prompt'));
            if (group && i > 1) group.style.display = i <= n ? '' : 'none';
        }
        const random = $('train_preview_custom_random');
        if (random) random.disabled = n < 2;
        const perPrompt = $('train_preview_custom_image_per_prompt');
        if (perPrompt) perPrompt.disabled = n < 2;
        const mode = $('train_preview_caption_mode');
        if (mode) {
            mode.disabled = n > 0;
            mode.style.opacity = n > 0 ? '0.5' : '';
            mode.title = n > 0 ? tr('Not used while Custom Prompts is above 0: the previews use your prompts.') : '';
        }
    }

    function build() {
        const container = $('custom_prompt_container');
        const first = $('train_preview_custom_prompt');
        const mode = $('train_preview_caption_mode');
        if (!container || !first || $('train_preview_custom_count')) return;

        // «Custom prompt» del Caption Mode pasa a ser Custom Prompts = 1.
        let wasCustom = false;
        if (mode) {
            const opt = mode.querySelector('option[value="custom"]');
            if (opt) {
                wasCustom = mode.value === 'custom';
                opt.remove();
                if (wasCustom) mode.value = 'first';
            }
        }

        // LTX y MiniMax-H3: el text encoder no cabe para codificar al vuelo, solo lo hace el Pre-Cache.
        const help = window.PREVIEW_PROMPTS_PRECACHE_ONLY
            ? 'Your own prompts for the previews. 0 = the previews use the dataset captions (Caption Mode). 1-4 = each preview round makes one image per prompt. This model encodes them only in Pre-Cache: after changing them, run Pre-Cache again (it skips the images already cached).'
            : 'Your own prompts for the previews. 0 = the previews use the dataset captions (Caption Mode). 1-4 = each preview round makes one image per prompt. They are encoded on GPU by Pre-Cache, or on CPU when you Save JSON during the training, and the trainer switches to them when they are ready.';
        const row = document.createElement('div');
        row.className = 'form-row';
        row.innerHTML = `
            <div class="form-group" title="${help}">
                <label>Custom Prompts (0-4):</label>
                <input type="number" id="train_preview_custom_count" value="${wasCustom ? 1 : 0}" min="0" max="${MAX}" step="1">
            </div>
            <div class="form-group" title="Each preview round makes a single image, with one of your prompts chosen at random, instead of one image per prompt. Previews take as long as with one prompt.">
                <label style="display: flex; align-items: center; gap: 8px; margin: 0; cursor: pointer;" for="train_preview_custom_random">
                    <input type="checkbox" id="train_preview_custom_random" style="width: 18px; height: 18px; cursor: pointer;">
                    One random prompt per preview
                </label>
            </div>`;
        container.parentNode.insertBefore(row, container);

        // Prompt 1 es la caja de siempre; la etiqueta pasa a «Prompt 1» y se añaden la 2, la 3 y la 4.
        const label = first.parentElement.querySelector('label');
        const wrap1 = document.createElement('div');
        wrap1.className = 'pp-prompt';
        first.parentElement.insertBefore(wrap1, label || first);
        if (label) {
            label.textContent = 'Prompt 1:';
            wrap1.appendChild(label);
        }
        wrap1.appendChild(first);
        let after = wrap1;
        for (let i = 2; i <= MAX; i++) {
            const wrap = document.createElement('div');
            wrap.className = 'pp-prompt';
            wrap.style.marginTop = '6px';
            const l = document.createElement('label');
            if (label) l.style.cssText = label.style.cssText;
            l.textContent = `Prompt ${i}:`;
            const box = document.createElement('textarea');
            box.id = promptId(i);
            box.rows = first.rows || 2;
            box.placeholder = first.getAttribute('placeholder') || '';
            wrap.append(l, box);
            after.after(wrap);
            after = wrap;
        }
        // Edición con varias imágenes de referencia (lista «+ Add image»): cada prompt puede editar la suya.
        const addImage = $('preview_edit_add_btn');
        if (addImage) {
            const opt = document.createElement('div');
            opt.style.marginTop = '6px';
            opt.title = 'Off: all the images are <image1>, <image2>... of one edit, for every prompt. On: each prompt edits its own image (prompt 1 -> image 1, prompt 2 -> image 2...; with fewer images than prompts they start over). Use it to see the effect on several photos in one preview round.';
            opt.innerHTML = `
                <label style="display: flex; align-items: center; gap: 8px; margin: 0; cursor: pointer;" for="train_preview_custom_image_per_prompt">
                    <input type="checkbox" id="train_preview_custom_image_per_prompt" style="width: 18px; height: 18px; cursor: pointer;">
                    One image per prompt
                </label>`;
            addImage.parentElement.appendChild(opt);
        }
        if (typeof I18N !== 'undefined') {
            I18N.apply(row);
            I18N.apply(container);
        }

        $('train_preview_custom_count').addEventListener('input', refresh);
        $('train_preview_custom_count').addEventListener('change', refresh);
        // Las páginas llaman a toggleCaptionMode al cambiar el selector y al cargar ajustes.
        window.toggleCaptionMode = () => refresh();

        if (typeof LIVE_APPLY !== 'undefined' && LIVE_APPLY.live) {
            LIVE_APPLY.live.preview_custom_count = 'train_preview_custom_count';
            LIVE_APPLY.live.preview_custom_random = 'train_preview_custom_random';
            if ($('train_preview_custom_image_per_prompt')) {
                LIVE_APPLY.live.preview_custom_image_per_prompt = 'train_preview_custom_image_per_prompt';
            }
            for (let i = 1; i <= MAX; i++) LIVE_APPLY.live[promptKey(i)] = promptId(i);
        }
        ready = true;
        refresh();
    }

    // Claves para el json (el prompt 1 ya lo envía cada página como preview_custom_prompt).
    function values() {
        if (!ready) return {};
        const out = {
            preview_custom_count: count(),
            preview_custom_random: !!($('train_preview_custom_random') || {}).checked,
        };
        if ($('train_preview_custom_image_per_prompt')) {
            out.preview_custom_image_per_prompt = $('train_preview_custom_image_per_prompt').checked;
        }
        for (let i = 2; i <= MAX; i++) out[promptKey(i)] = ($(promptId(i)) || { value: '' }).value.trim();
        return out;
    }

    // Pone los valores de unos ajustes cargados. Un json antiguo con Caption Mode = custom es 1 prompt.
    function load(data) {
        if (!ready || !data) return;
        const countEl = $('train_preview_custom_count');
        const mode = $('train_preview_caption_mode');
        if (data.preview_custom_count !== undefined && data.preview_custom_count !== null) {
            countEl.value = Math.max(0, Math.min(MAX, parseInt(data.preview_custom_count, 10) || 0));
        } else if (data.preview_caption_mode === 'custom') {
            countEl.value = 1;
        } else if (data.preview_caption_mode !== undefined) {
            countEl.value = 0;
        }
        if (mode && (data.preview_caption_mode === 'custom' || !mode.value)) mode.value = 'first';
        if (data.preview_custom_random !== undefined) $('train_preview_custom_random').checked = !!data.preview_custom_random;
        const perPrompt = $('train_preview_custom_image_per_prompt');
        if (perPrompt && data.preview_custom_image_per_prompt !== undefined) perPrompt.checked = !!data.preview_custom_image_per_prompt;
        for (let i = 2; i <= MAX; i++) {
            if (data[promptKey(i)] !== undefined) $(promptId(i)).value = data[promptKey(i)] || '';
        }
        refresh();
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build);
    else build();

    return { values, load, refresh };
})();
