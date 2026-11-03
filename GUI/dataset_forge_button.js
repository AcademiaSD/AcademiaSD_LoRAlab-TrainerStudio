// dataset_forge_button.js — Botón «🧬 Dataset Forge» en la barra del Dataset Manager de cada entrenador.
// Dataset Forge button in the Dataset Manager toolbar of every trainer.
//
// Abre /forge (scripts/dataset_forge.py) en otra pestaña. Cuando el Forge termina y se pulsa «Usar en el
// Dataset Manager», llega la carpeta nueva (BroadcastChannel o postMessage, mismo origen) y se pone como
// carpeta del dataset: el evento change hace lo mismo que escribirla a mano (guardar ajustes y refrescar).
(function () {
    const tr = (k, v) => (typeof t === 'function' ? t(k, v) : k);

    function build() {
        const bar = document.querySelector('.ds-toolbar');
        if (!bar || document.getElementById('btn-ds-forge')) return;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.id = 'btn-ds-forge';
        btn.className = 'btn btn-primary btn-sm';
        btn.style.background = 'linear-gradient(90deg, #0e7490, #6d28d9)';
        btn.textContent = '🧬 Dataset Forge';
        btn.title = tr('Creates a character dataset from a few images (from 1): new outfits, places and framings that keep the face.');
        btn.addEventListener('click', () => window.open('/forge', '_blank'));
        bar.insertBefore(btn, bar.firstChild);
    }

    function useFolder(path) {
        const input = document.getElementById('dataset_path');
        // Llega dos veces (BroadcastChannel y opener): la segunda ya no cambia nada.
        if (!input || !path || input.value === path) return;
        input.value = path;
        input.dispatchEvent(new Event('change', { bubbles: true }));
        if (typeof loadDatasetInfo === 'function') setTimeout(loadDatasetInfo, 600);
        if (typeof toast === 'function') toast('🧬 ' + tr('Dataset folder: {path}', { path }), 'success');
    }

    const onMessage = data => { if (data && data.type === 'loralab-forge-done') useFolder(data.path); };
    try { new BroadcastChannel('loralab-forge').onmessage = e => onMessage(e.data); } catch (e) {}
    window.addEventListener('message', e => { if (e.origin === location.origin) onMessage(e.data); });

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build);
    else build();
})();
