// file_transfer.js — Subir el dataset y descargar los LoRAs desde el navegador (scripts/file_transfer.py).
// Upload the dataset and download the LoRAs from the browser. Shared by every trainer UI.

function ftUpload() {
    const input = document.getElementById('ds-upload');
    input.value = '';
    input.click();
}

function ftSend(files) {
    if (!files || !files.length) return;
    const btn = document.getElementById('btn-ds-upload');
    const label = btn.innerText;
    const form = new FormData();
    for (const f of files) form.append('files', f, f.name);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload-dataset');
    xhr.upload.onprogress = e => { if (e.lengthComputable) btn.innerText = `⬆ ${Math.round(e.loaded / e.total * 100)}%`; };
    xhr.onloadend = () => {
        btn.innerText = label;
        btn.disabled = false;
        let data = null;
        try { data = JSON.parse(xhr.responseText); } catch (e) { }
        if (!data || data.status !== 'ok') {
            alert(t('Upload failed') + ': ' + (data && data.error || xhr.status));
            return;
        }
        let msg = '✓ ' + t('{n} file(s) uploaded', { n: data.saved }) + ` → ${data.path}`;
        if (data.skipped.length) msg += '\n\n' + t('Skipped (unsupported type):') + `\n${data.skipped.join('\n')}`;
        alert(msg);
        if (typeof loadDatasetInfo === 'function') loadDatasetInfo();
    };
    btn.disabled = true;
    xhr.send(form);
}

async function ftDownload() {
    const data = await (await fetch('/api/output-files')).json();
    let box = document.getElementById('ft-modal');
    if (!box) {
        box = document.createElement('div');
        box.id = 'ft-modal';
        box.style.cssText = 'position:fixed;inset:0;display:flex;align-items:center;justify-content:center;background:rgba(2,6,23,0.75);z-index:1000;';
        box.onclick = e => { if (e.target === box) box.remove(); };
        document.body.appendChild(box);
    }
    const rows = data.files.length
        ? data.files.map(f => `<a href="/api/download-output/${encodeURIComponent(f.name)}" download
              style="display:flex;justify-content:space-between;gap:16px;padding:8px 10px;border-radius:6px;color:var(--text-main);text-decoration:none;background:var(--bg-dark);border:1px solid var(--panel-border);">
              <span>⬇ ${f.name}</span><span style="color:var(--text-muted);">${(f.size / 1048576).toFixed(1)} MB</span></a>`).join('')
        : `<div style="color:var(--text-muted);">${t('No LoRA yet.')}</div>`;
    box.innerHTML = `<div style="width:min(560px,92vw);max-height:80vh;overflow:auto;display:flex;flex-direction:column;gap:8px;padding:20px;
            background:var(--panel-bg);border:1px solid var(--panel-border);border-radius:12px;">
        <div style="font-weight:700;">${t('⬇ Download LoRA')}</div>
        <div style="font-size:0.78rem;color:var(--text-muted);word-break:break-all;">${data.path}</div>
        ${rows}
        <button type="button" class="btn btn-secondary btn-sm" onclick="document.getElementById('ft-modal').remove()">${t('Close')}</button>
    </div>`;
}

// Arrastrar archivos o un .zip sobre el Dataset Manager los sube a la carpeta del dataset.
window.addEventListener('DOMContentLoaded', () => {
    document.getElementById('ds-upload').onchange = e => ftSend(e.target.files);
    const grid = document.getElementById('ds-grid');
    const zone = grid.parentElement;
    zone.addEventListener('dragover', e => {
        if (!e.dataTransfer.types.includes('Files')) return;
        e.preventDefault();
        grid.style.outline = '2px dashed var(--accent)';
    });
    zone.addEventListener('dragleave', e => { if (!zone.contains(e.relatedTarget)) grid.style.outline = ''; });
    zone.addEventListener('drop', e => {
        if (!e.dataTransfer.files.length) return;
        e.preventDefault();
        grid.style.outline = '';
        ftSend(e.dataTransfer.files);
    });
});
