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

    // Al acabar solo interesan los fallos: las filas ✓ estorban para ver qué hay que reintentar.
    // When the batch ends only the failures matter: the ✓ rows just hide what needs a retry.
    const box = ftStatusBox();
    box.querySelectorAll('.ft-row.ft-ok, .ft-row.ft-pending, .ft-summary').forEach(el => el.remove());
    if (!ftFailedFiles.length) {
        box.style.display = 'none';
        box.innerHTML = '';
    } else {
        const sum = document.createElement('div');
        sum.className = 'ft-summary';
        sum.innerText = t('{n} file(s) failed to upload', { n: ftFailedFiles.length });
        box.insertBefore(sum, box.firstChild);
    }
    ftRetryButton();

    if (typeof loadDatasetInfo === 'function') loadDatasetInfo();

    if (ftFailedFiles.length) {
        toast(t('{ok} uploaded, {n} failed. Only the failed files are left, with Retry failed.',
                { ok: ok, n: ftFailedFiles.length }), ok ? 'warning' : 'error');
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
        const style = document.createElement('style');
        style.textContent = '#ft-upload-status .ft-row{display:flex;gap:8px;padding:2px 0;color:var(--text-muted);}'
            + ' #ft-upload-status .ft-row.ft-ok{color:var(--success);}'
            + ' #ft-upload-status .ft-row.ft-pending{opacity:.85;}'
            + ' #ft-upload-status .ft-row.ft-fail{background:rgba(239,68,68,.16);border-left:3px solid var(--danger);'
            + 'border-radius:4px;padding:4px 6px;color:#fecaca;font-weight:600;}'
            + ' #ft-upload-status .ft-summary{font-weight:700;color:#fca5a5;padding:2px 0 6px 0;}';
        document.head.appendChild(style);
        box = document.createElement('div');
        box.id = 'ft-upload-status';
        box.style.cssText = 'display:none;margin-top:8px;max-height:220px;overflow:auto;font-size:0.82rem;'
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
        row.innerHTML = '<span class="ft-icon"></span>'
            + '<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"></span>'
            + '<span class="ft-detail"></span>';
        box.appendChild(row);
    }
    row.className = `ft-row ft-${cls}`;
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

function ftEscape(value) {
    return String(value).replace(/[&<>"']/g, ch => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[ch]));
}

async function ftDownload() {
    // 'Final LoRA Filename' (ya lleva el Project Name): el servidor lo usa para nombrar la
    // descarga del final y de los checkpoints (_<pasos>_steps). No cambia nada en disco.
    // 'Final LoRA Filename' (Project Name already inside): the server uses it to name the
    // download of the final LoRA and the checkpoints (_<steps>_steps). Nothing on disk changes.
    const finalName = (document.getElementById('export_final_name')?.value || '').trim();
    const data = await (await fetch('/api/output-files?final_name=' + encodeURIComponent(finalName))).json();
    let box = document.getElementById('ft-modal');
    if (!box) {
        box = document.createElement('div');
        box.id = 'ft-modal';
        box.style.cssText = 'position:fixed;inset:0;display:flex;align-items:center;justify-content:center;background:rgba(2,6,23,0.75);z-index:1000;';
        box.onclick = e => { if (e.target === box) box.remove(); };
        document.body.appendChild(box);
    }
    const hf = await (await fetch('/api/hf-upload-settings')).json();
    const rows = data.files.length
        ? data.files.map(f => `<div style="display:flex;gap:6px;align-items:stretch;min-width:0;">
            <a href="/api/download-output/${encodeURIComponent(f.name)}?final_name=${encodeURIComponent(finalName)}" download
              style="flex:1;display:flex;justify-content:space-between;align-items:center;gap:16px;padding:8px 10px;border-radius:6px;color:var(--text-main);text-decoration:none;background:var(--bg-dark);border:1px solid var(--panel-border);min-width:0;">
              <span title="${ftEscape(f.download_name || f.name)}" style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">⬇ ${ftEscape(f.download_name || f.name)}</span><span style="flex:0 0 auto;white-space:nowrap;color:var(--text-muted);">${(f.size / 1048576).toFixed(1)} MB</span></a>
            <button type="button" class="btn btn-secondary btn-sm" data-name="${ftEscape(f.name)}" onclick="ftHfUpload(this)"
                    title="${t('Upload to Hugging Face')}" style="flex:0 0 auto;">☁</button></div>`).join('')
        : `<div style="color:var(--text-muted);">${t('No LoRA yet.')}</div>`;
    box.innerHTML = `<div style="width:min(560px,92vw);max-height:80vh;overflow:auto;display:flex;flex-direction:column;gap:8px;padding:20px;
            background:var(--panel-bg);border:1px solid var(--panel-border);border-radius:12px;">
        <div style="font-weight:700;">${t('⬇ Download LoRA')}</div>
        <div style="font-size:0.78rem;color:var(--text-muted);word-break:break-all;">${ftEscape(data.path)}</div>
        ${rows}
        <div style="display:flex;flex-direction:column;gap:6px;margin-top:6px;padding-top:10px;border-top:1px solid var(--panel-border);">
            <div style="font-weight:700;">${t('☁ Upload to Hugging Face')}</div>
            ${hf.has_token ? '' : `<div style="font-size:0.78rem;color:var(--warning);">${t('Save a Hugging Face token with write access in the Hugging Face Token panel first.')}</div>`}
            <input type="text" id="ft-hf-repo" value="${ftEscape(hf.repo)}" placeholder="user/my-loras" autocomplete="off" onchange="ftHfSave()">
            <label style="display:flex;gap:6px;align-items:center;font-size:0.85rem;"><input type="checkbox" id="ft-hf-private" ${hf.private ? 'checked' : ''} onchange="ftHfSave()"> ${t('Private repo (when it is created)')}</label>
            <label style="display:flex;gap:6px;align-items:center;font-size:0.85rem;" title="${t('Every LoRA the trainer saves from now on (checkpoints and final) goes up to the repo by itself. Older ones: use the ☁ button.')}">
                <input type="checkbox" id="ft-hf-auto" ${hf.auto ? 'checked' : ''} onchange="ftHfSave()"> ${t('Upload new LoRAs automatically')}</label>
            ${hf.last ? `<div style="font-size:0.78rem;color:var(--text-muted);word-break:break-all;">${ftEscape(hf.last)}</div>` : ''}
        </div>
        <button type="button" class="btn btn-secondary btn-sm" onclick="document.getElementById('ft-modal').remove()">${t('Close')}</button>
    </div>`;
}

function ftHfFields() {
    return {
        repo: document.getElementById('ft-hf-repo').value.trim(),
        private: document.getElementById('ft-hf-private').checked,
        auto: document.getElementById('ft-hf-auto').checked,
        final_name: (document.getElementById('export_final_name')?.value || '').trim(),
    };
}

async function ftHfSave() {
    const fields = ftHfFields();
    const res = await fetch('/api/save-hf-upload-settings', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(fields)
    });
    const data = await res.json();
    if (data.status !== 'ok') { toast(data.error || `HTTP ${res.status}`, 'error'); return; }
    if (fields.auto && !data.auto) toast(t('Write the repo (user/name) to upload automatically.'), 'warning');
    document.getElementById('ft-hf-auto').checked = data.auto;
}

// Un LoRA al repo; el botón muestra el progreso y, al acabar, abre el archivo en Hugging Face.
// One LoRA to the repo; the button shows progress and, once done, links to the file on Hugging Face.
async function ftHfUpload(btn) {
    const fields = ftHfFields();
    if (!fields.repo) { toast(t('Write the repo (user/name) first.'), 'warning'); return; }
    btn.disabled = true;
    btn.innerText = '…';
    try {
        const res = await fetch('/api/hf-upload', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ...fields, name: btn.dataset.name })
        });
        const data = await res.json();
        if (data.status !== 'ok') throw new Error(data.error || `HTTP ${res.status}`);
        btn.innerText = '✓';
        btn.onclick = () => window.open(data.url, '_blank');
        btn.disabled = false;
        toast(t('Uploaded to Hugging Face'), 'success');
    } catch (err) {
        btn.innerText = '☁';
        btn.disabled = false;
        toast(String(err.message || err), 'error');
    }
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
