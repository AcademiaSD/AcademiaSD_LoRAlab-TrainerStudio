// Preview aspect ratio + Preview Size (area).
// Size keeps the classic area semantics: 1024 means ~1024 x 1024 px², not a
// 1024 px long edge. Dimensions are searched among multiples of the model step
// near the ideal width/height and clamped to maxSide, keeping area and ratio as
// close as the model constraint allows.
// "Training" (size 0) keeps the legacy trainer behaviour untouched.
(function () {
    const RATIOS = {
        '1:1': [1, 1],
        '2:3': [2, 3],
        '3:4': [3, 4],
        '4:3': [4, 3],
        '16:9': [16, 9],
        '9:16': [9, 16],
    };

    const config = { step: 32, maxSide: 0 };

    function configure(options) {
        if (!options) return;
        if (Number.isFinite(options.step) && options.step >= 1) {
            config.step = Math.round(options.step);
        }
        if (Number.isFinite(options.maxSide) && options.maxSide >= 1) {
            config.maxSide = Math.round(options.maxSide);
        }
    }

    function compute(aspect, size, step = config.step, maxSide = config.maxSide) {
        const ratio = RATIOS[aspect];
        if (!ratio || !size) return { width: 0, height: 0 };
        const rw = ratio[0];
        const rh = ratio[1];
        const ratioValue = rw / rh;
        const area = size * size;
        const stepPx = Math.max(1, Math.round(step || 32));
        const limit = maxSide > 0 ? Math.round(maxSide) : 0;
        const snap = value => Math.max(stepPx, Math.round(value / stepPx) * stepPx);

        const idealW = snap(Math.sqrt(area * ratioValue));
        const idealH = snap(Math.sqrt(area / ratioValue));
        const widths = new Set();
        const heights = new Set();
        for (let i = -3; i <= 3; i++) {
            const width = idealW + i * stepPx;
            const height = idealH + i * stepPx;
            if (width >= stepPx && (!limit || width <= limit)) widths.add(width);
            if (height >= stepPx && (!limit || height <= limit)) heights.add(height);
        }

        let best = null;
        for (const width of widths) {
            for (const height of heights) {
                const score = Math.abs(width * height - area) / area
                    + Math.abs(width / height - ratioValue) / ratioValue;
                if (!best || score < best.score) {
                    best = { width, height, score };
                }
            }
        }

        if (best) return { width: best.width, height: best.height };

        let width = idealW;
        let height = idealH;
        if (limit && Math.max(width, height) > limit) {
            const scale = limit / Math.max(width, height);
            width = Math.max(stepPx, Math.floor(width * scale / stepPx) * stepPx);
            height = Math.max(stepPx, Math.floor(height * scale / stepPx) * stepPx);
        }
        return { width, height };
    }

    function controls() {
        return {
            aspect: document.getElementById('train_preview_aspect'),
            size: document.getElementById('train_preview_size'),
            hint: document.getElementById('preview_size_hint'),
        };
    }

    function sync() {
        const { aspect, size, hint } = controls();
        if (!aspect || !size) return;
        if (aspect.value !== 'auto' && (parseInt(size.value, 10) || 0) === 0) {
            size.value = '1024';
        }
        const sizeValue = parseInt(size.value, 10) || 0;
        const dim = aspect.value !== 'auto' ? compute(aspect.value, sizeValue) : { width: 0, height: 0 };
        if (!hint) return;
        if (dim.width > 0) {
            hint.textContent = `${dim.width} × ${dim.height} px · ${(dim.width * dim.height).toLocaleString()} px²`;
        } else if (sizeValue > 0) {
            hint.textContent = `${sizeValue} × ${sizeValue} px² (${t('area')}) · ${t('sample ratio')}`;
        } else {
            hint.textContent = t('Training sample size and aspect ratio');
        }
    }

    function values() {
        const { aspect, size } = controls();
        const aspectValue = aspect ? aspect.value : 'auto';
        const sizeValue = size ? (parseInt(size.value, 10) || 0) : 0;
        const dim = aspectValue !== 'auto' ? compute(aspectValue, sizeValue) : { width: 0, height: 0 };
        return {
            preview_size: sizeValue,
            preview_aspect_ratio: aspectValue,
            preview_width: dim.width,
            preview_height: dim.height,
        };
    }

    function apply(data) {
        const { aspect, size } = controls();
        if (!data) return;
        if (aspect) {
            aspect.value = data.preview_aspect_ratio !== undefined ? String(data.preview_aspect_ratio) : 'auto';
            if (!RATIOS[aspect.value] && aspect.value !== 'auto') aspect.value = 'auto';
        }
        if (size) {
            size.value = String(data.preview_size !== undefined ? (data.preview_size || 0) : 0);
        }
        sync();
    }

    window.PreviewSize = { configure, values, apply, compute, sync };
    window.syncPreviewSize = sync;
    document.addEventListener('DOMContentLoaded', sync);
    window.addEventListener('load', sync);
})();
