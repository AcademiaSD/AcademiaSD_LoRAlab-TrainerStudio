// slider_text.js — Slider LoRA por texto, común a los trainers.
// Text slider LoRA, shared by the trainers.
//
// Pone su casilla en la rejilla de opciones (debajo del LR Scheduler) y, con ella marcada, los dos extremos
// del concepto, la intensidad y las imágenes de práctica. Se guardan como slider_text / slider_pos /
// slider_neg / slider_eta / slider_bank / slider_ultra por el mismo camino que los Custom Prompts: el
// Pre-Cache codifica los extremos y el trainer entrena el dial. La lógica está en scripts/slider_text.py.
// Cada trainer da sus valores por defecto en window.LoRAlabSliderDefaults antes de cargar este archivo
// ({eta, bank, ultra}; sin ultra no se muestra esa casilla).
(function () {
    const $ = (id) => document.getElementById(id);
    const D = Object.assign({ eta: 1.5, bank: 0 }, window.LoRAlabSliderDefaults || {});
    const IDS = { on: 'train_slider_text', pos: 'train_slider_pos', neg: 'train_slider_neg', eta: 'train_slider_eta',
                  bank: 'train_slider_bank', ultra: 'train_slider_ultra' };

    function update() {
        const on = $(IDS.on)?.checked;
        if (!$('slider_text_fields')) return;
        $('slider_text_fields').style.display = on ? '' : 'none';
    }

    function build() {
        const sched = $('train_lr_scheduler');
        if (!sched || $(IDS.on) || !window.LoRAlabChecks) return;
        const holder = document.createElement('div');
        holder.innerHTML = `<div class="form-group" title="Trains a dial instead of a subject or a style: the LoRA strength moves the picture from the negative end to the positive one (for example from very thin to very fat). Strength 0 is the model as it is and every point of strength moves it the same amount, so +2 is a little and +5 is the end. It needs no before and after pictures: the base model already knows both ends, and the dataset captions say what the pictures show. Run the Pre-Cache after changing the ends.">
                    <label style="display: flex; align-items: center; gap: 8px; margin: 0; cursor: pointer;" for="${IDS.on}">
                        <input type="checkbox" id="${IDS.on}" style="width: 18px; height: 18px; cursor: pointer;">
                        Text slider (dial)
                    </label>
                </div>`;
        const checks = window.LoRAlabChecks.place(holder.firstElementChild, sched.closest('.form-row') || sched.parentElement);
        const anchor = $('masked_loss_fields') || $('preservation_hint') || window.LoRAlabChecks.last() || checks;
        const ultra = D.ultra === undefined ? '' : `
                <div class="form-group" style="display: flex; align-items: flex-end;" title="Ultra: the LoRA only goes into the composition blocks. The dial holds up at much higher strengths and changes the shape more than the details.">
                    <label style="display: flex; align-items: center; gap: 8px; margin: 0 0 10px; cursor: pointer;" for="${IDS.ultra}">
                        <input type="checkbox" id="${IDS.ultra}" style="width: 18px; height: 18px; cursor: pointer;" ${D.ultra ? 'checked' : ''}>
                        Ultra (composition blocks)
                    </label>
                </div>`;
        const box = document.createElement('div');
        box.innerHTML = `
            <div id="slider_text_fields" style="display: none;">
                <div class="form-row">
                    <div class="form-group" title="What strength +5 adds. Describe the body or the trait, not the face or the camera: a face word can move the camera closer. Example: a very fat obese person, overweight all over the body: chubby face, thick arms, thick thighs, wide waist and hips">
                        <label for="${IDS.pos}">Positive end (+):</label>
                        <input type="text" id="${IDS.pos}" placeholder="a very fat obese person, overweight all over the body">
                    </div>
                    <div class="form-group" title="What strength -5 adds, the opposite of the positive end and written the same way. Example: a very thin skinny underweight person, slim all over the body: thin arms, thin legs, narrow waist and hips">
                        <label for="${IDS.neg}">Negative end (−):</label>
                        <input type="text" id="${IDS.neg}" placeholder="a very thin skinny underweight person, slim all over the body">
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group" title="How far strength 5 goes, as a multiple of the difference the base model makes between the two ends. Raise it if the dial changes too little, lower it if the ends break the picture.">
                        <label for="${IDS.eta}">Slider intensity:</label>
                        <input type="number" id="${IDS.eta}" value="${D.eta}" min="0.5" max="20" step="0.5">
                    </div>
                    <div class="form-group" title="Before training, the base model draws this many pictures from the dataset captions and the slider learns on them instead of on the dataset photos: the dial then keeps the framing the model gives to each prompt. 0 = the dataset photos.">
                        <label for="${IDS.bank}">Practice images:</label>
                        <input type="number" id="${IDS.bank}" value="${D.bank}" min="0" max="256" step="1">
                    </div>${ultra}
                </div>
                <div id="slider_text_hint" style="font-size: 0.78rem; color: var(--text-muted, #94a3b8); line-height: 1.35; margin: -4px 0 10px;">Strength 0 is the model as it is; +5 and −5 in ComfyUI are the two ends. The previews show the dial. Run the Pre-Cache after changing the ends.</div>
            </div>`;
        anchor.after(...box.children);
        if (typeof I18N !== 'undefined') { I18N.apply($('slider_text_fields')); I18N.apply($(IDS.on).closest('.form-row')); }
        $(IDS.on).addEventListener('change', update);
        if (typeof LIVE_APPLY !== 'undefined' && LIVE_APPLY.new) {
            Object.assign(LIVE_APPLY.new, { slider_text: IDS.on, slider_pos: IDS.pos, slider_neg: IDS.neg,
                                            slider_eta: IDS.eta, slider_bank: IDS.bank });
            if ($(IDS.ultra)) LIVE_APPLY.new.slider_ultra = IDS.ultra;
        }
        update();
    }

    if (typeof PreviewPrompts !== 'undefined') {
        PreviewPrompts.register({
            values: () => {
                if (!$(IDS.on)) return {};
                const num = (id, def, lo, hi) => (v => (isNaN(v) ? def : Math.max(lo, Math.min(hi, v))))(parseFloat($(id).value));
                const out = {
                    slider_text: $(IDS.on).checked,
                    slider_pos: $(IDS.pos).value.trim(),
                    slider_neg: $(IDS.neg).value.trim(),
                    slider_eta: num(IDS.eta, D.eta, 0.5, 20),
                    slider_bank: Math.round(num(IDS.bank, D.bank, 0, 256)),
                };
                if ($(IDS.ultra)) out.slider_ultra = $(IDS.ultra).checked;
                return out;
            },
            load: (data) => {
                if (!$(IDS.on)) return;
                $(IDS.on).checked = !!data.slider_text;
                if (data.slider_pos !== undefined) $(IDS.pos).value = data.slider_pos;
                if (data.slider_neg !== undefined) $(IDS.neg).value = data.slider_neg;
                if (data.slider_eta !== undefined) $(IDS.eta).value = data.slider_eta;
                if (data.slider_bank !== undefined) $(IDS.bank).value = data.slider_bank;
                if ($(IDS.ultra) && data.slider_ultra !== undefined) $(IDS.ultra).checked = !!data.slider_ultra;
                update();
            },
        });
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build);
    else build();
})();
