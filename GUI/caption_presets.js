// caption_presets.js — Tipo de captions (General / Character) junto al prompt del captioner, común a los
// trainers de imagen. Caption type (General / Character) next to the captioner prompt.
//
// General: el prompt de siempre (descripción larga: apariencia, pelo, ropa, pose, fondo, luz), bueno para
// estilos y objetos. Character: captions cortos sin los rasgos fijos de la persona (cara, pelo, ojos, piel,
// edad), solo lo que cambia entre fotos; así la identidad queda en el trigger, se parece más con prompts
// cualquiera y casa con la protección de salida (la palabra de clase sustituye al trigger). Solo cambia el
// texto del prompt, que se guarda como siempre; solo aparece en el modo de descripción normal.
(function () {
    const CHARACTER = 'Describe this photo in one or two sentences, at most 40 words, for training a character LoRA. '
        + "Refer to the person only as 'a woman', 'a man' or 'a person'. Do not describe their fixed traits: "
        + 'face, hair colour or length, eye colour, skin, age or body. Describe only what changes between photos: '
        + 'clothing, pose and hands, expression, framing and camera angle, background and lighting. '
        + 'No preamble, and do not mention that this is an image or a photo.';
    const $ = (id) => document.getElementById(id);
    let general = null;           // el prompt general que había antes de elegir Character

    function textarea() { return $('caption_prompt_text'); }
    function modeSelect() { return $('caption_mode') || $('caption_style'); }
    function describing() {
        const m = modeSelect();
        return !m || m.value === 'normal' || m.value === 'natural';
    }
    function isCharacter() { return (textarea()?.value || '').trim() === CHARACTER; }

    function trainerDefault() {
        try { if (typeof DEFAULT_CAPTION_PROMPT === 'string') return DEFAULT_CAPTION_PROMPT; } catch (e) {}
        try { if (DEFAULT_CAPTION_PROMPTS && DEFAULT_CAPTION_PROMPTS.normal) return DEFAULT_CAPTION_PROMPTS.normal; } catch (e) {}
        try { if (DEFAULT_CAPTION_PROMPTS && DEFAULT_CAPTION_PROMPTS.natural) return DEFAULT_CAPTION_PROMPTS.natural; } catch (e) {}
        return '';
    }

    function sync() {
        const sel = $('caption_preset');
        if (!sel) return;
        sel.style.display = describing() ? '' : 'none';
        sel.value = isCharacter() ? 'character' : 'general';
    }

    function choose() {
        const ta = textarea();
        if ($('caption_preset').value === 'character') {
            if (!isCharacter()) general = ta.value;
            ta.value = CHARACTER;
        } else if (isCharacter()) {
            ta.value = general || trainerDefault();
        }
        ta.dispatchEvent(new Event('input', { bubbles: true }));
        ta.dispatchEvent(new Event('change', { bubbles: true }));
    }

    function build() {
        const ta = textarea();
        if (!ta || $('caption_preset')) return;
        const sel = document.createElement('select');
        sel.id = 'caption_preset';
        sel.className = 'ds-mode-select';
        sel.title = 'General: a long description (appearance, hair, clothing, pose, background, lighting), for styles and objects. '
            + 'Character: short captions without the fixed traits of the person (face, hair, eyes, skin, age), only what changes '
            + 'between photos, so the trigger learns the identity; best for people, and with Output preservation.';
        sel.innerHTML = '<option value="general">General</option><option value="character">Character</option>';
        ta.before(sel);
        if (typeof I18N !== 'undefined') I18N.apply(sel.parentElement);
        sel.addEventListener('change', choose);
        ta.addEventListener('input', sync);
        modeSelect()?.addEventListener('change', () => setTimeout(sync, 0));
        sync();
        // Los ajustes del captioner se cargan después: se sigue el texto del prompt.
        setInterval(sync, 1500);
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build);
    else build();
})();
