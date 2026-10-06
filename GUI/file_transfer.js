// file_transfer.js — Subir el dataset y descargar los LoRAs desde el navegador (scripts/file_transfer.py).
// Upload the dataset and download the LoRAs from the browser. Shared by every trainer UI.

function ftUpload() {
    const input = document.getElementById('ds-upload');
    input.value = '';
    input.click();
}

// Subida por archivo: un fallo no cancela el resto, cada archivo muestra ✓/✗ y se puede
// reintentar solo lo fallido. Los avisos usan los toasts de /notify.js.
let ftFailedFiles = [];

function ftSend(files) {
    const list = [...(files || [])];
    if (!list.length) return;
    ftStatusBox().style.display = 'block';
    ftRunQueue(list);
}

async function ftRunQueue(files) {
    const btn = document.getElementById('btn-ds-upload');
    const label = btn.innerText;
    let ok = 0;
    ftFailedFiles = [];

    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        btn.disabled = true;
        btn.innerText = `⬆ ${i + 1}/${files.length}`;
        ftRow(file, '…', t('Uploading {i}/{n}', { i: i + 1, n: files.length }), 'pending');
        try {
            const data = await ftPostOne(file);
            const bad = (data.failed || []).find(f => f.name === file.name);
            if (bad) throw new Error(bad.error);
            if (data.status !== 'ok' && data.status !== 'partial') throw new Error(data.error || `HTTP ${data.http}`);
            ftRow(file, '✓', t('Uploaded'), 'ok');
            ok++;
        } catch (err) {
            ftRow(file, '✗', String(err.message || err), 'fail');
            ftFailedFiles.push(file);
        }
    }

    btn.disabled = false;
    btn.innerText = label;
    ftRetryButton();

    if (typeof loadDatasetInfo === 'function') loadDatasetInfo();

    if (ftFailedFiles.length) {
        toast(t('{ok} uploaded, {n} failed. Use Retry failed.', { ok: ok, n: ftFailedFiles.length }), 'warning');
    } else if (files.length) {
        toast(t('{n} file(s) uploaded', { n: ok }), 'success');
    }
}

function ftPostOne(file) {
    return new Promise((resolve, reject) => {
        const form = new FormData();
        form.append('files', file, file.name);
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/api/upload-dataset');
        xhr.onloadend = () => {
            let data = null;
            try { data = JSON.parse(xhr.responseText); } catch (e) { }
            if (!data) { reject(new Error(`HTTP ${xhr.status}`)); return; }
            data.http = xhr.status;
            resolve(data);
        };
        xhr.send(form);
    });
}

// --- Panel de estado por archivo -------------------------------------------------
function ftStatusBox() {
    let box = document.getElementById('ft-upload-status');
    if (!box) {
        box = document.createElement('div');
        box.id = 'ft-upload-status';
        box.style.cssText = 'display:none;margin-top:8px;max-height:180px;overflow:auto;font-size:0.82rem;'
            + 'background:var(--bg-dark);border:1px solid var(--panel-border);border-radius:8px;padding:8px;';
        const grid = document.getElementById('ds-grid');
        grid.parentElement.insertBefore(box, grid.nextSibling);
    }
    return box;
}

function ftRow(file, icon, text, cls) {
    const box = ftStatusBox();
    let row = box.querySelector(`[data-file="${CSS.escape(file.name)}"]`);
    if (!row) {
        row = document.createElement('div');
        row.dataset.file = file.name;
        row.style.cssText = 'display:flex;gap:8px;padding:2px 0;color:var(--text-muted);';
        row.innerHTML = '<span class="ft-icon"></span>'
            + '<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"></span>'
            + '<span class="ft-detail"></span>';
        box.appendChild(row);
    }
    row.className = `ft-${cls}`;
    row.querySelector('.ft-icon').innerText = icon;
    row.querySelector('span:nth-child(2)').innerText = file.name;
    row.querySelector('.ft-detail').innerText = text;
}

function ftRetryButton() {
    const box = ftStatusBox();
    let btn = document.getElementById('ft-retry');
    if (!btn) {
        btn = document.createElement('button');
        btn.id = 'ft-retry';
        btn.type = 'button';
        btn.className = 'btn btn-warning btn-sm';
        box.appendChild(btn);
    }
    btn.innerText = t('Retry failed files');
    btn.style.display = ftFailedFiles.length ? '' : 'none';
    btn.onclick = () => {
        const again = ftFailedFiles.slice();
        if (again.length) ftSend(again);
    };
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
              <span>⬇ ${f.download_name || f.name}</span><span style="color:var(--text-muted);">${(f.size / 1048576).toFixed(1)} MB</span></a>`).join('')
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
