// notify.js — Avisos y confirmaciones propios, compartidos por todas las GUI.
// Sustituyen a alert()/confirm() del navegador: no congelan la página y encajan con el tema.

const NOTIFY_CSS = `
/* Centrados arriba: el aviso cae donde ya está la mirada al trabajar el panel, sin
   recorrer la pantalla hasta la esquina. Center-top: the notice lands where the eye
   already is while working the panel, instead of a trip to the corner. */
#toast-stack { position: fixed; top: 16px; left: 50%; transform: translateX(-50%); z-index: 3000;
    display: flex; flex-direction: column; align-items: center; gap: 8px; max-width: min(420px, 90vw); }
.toast { display: flex; align-items: flex-start; gap: 10px; padding: 10px 12px; border-radius: 8px;
    background: var(--panel-bg, #1e293b); border: 1px solid var(--panel-border, #334155);
    color: var(--text-main, #f8fafc); font-size: 0.85rem; white-space: pre-line;
    word-break: break-word; box-shadow: 0 8px 24px rgba(0,0,0,0.45); animation: toast-in 0.18s ease-out; }
.toast.success { border-color: #16a34a; }
.toast.warning { border-color: #d97706; }
.toast.error   { border-color: #dc2626; }
.toast .toast-close { cursor: pointer; opacity: 0.6; border: 0; background: none;
    color: inherit; font-size: 1rem; line-height: 1; }
.toast .toast-close:hover { opacity: 1; }
@keyframes toast-in { from { transform: translateY(-8px); opacity: 0; } to { transform: none; opacity: 1; } }
#ask-modal { position: fixed; inset: 0; z-index: 3100; display: none; align-items: center;
    justify-content: center; background: rgba(2,6,23,0.75); }
#ask-modal.active { display: flex; }
#ask-modal .ask-box { width: min(520px, 92vw); display: flex; flex-direction: column; gap: 14px;
    background: var(--panel-bg, #1e293b); border: 1px solid var(--panel-border, #334155);
    border-radius: 12px; padding: 18px; color: var(--text-main, #f8fafc); font-size: 0.9rem;
    white-space: pre-line; }
#ask-modal .ask-actions { display: flex; justify-content: flex-end; gap: 8px; }
`;

function _notifyInit() {
    if (document.getElementById('toast-stack')) return;
    const style = document.createElement('style');
    style.textContent = NOTIFY_CSS;
    document.head.appendChild(style);
    const stack = document.createElement('div');
    stack.id = 'toast-stack';
    document.body.appendChild(stack);
    const modal = document.createElement('div');
    modal.id = 'ask-modal';
    modal.innerHTML = '<div class="ask-box"><div class="ask-text"></div>'
        + '<div class="ask-actions">'
        + '<button type="button" class="btn btn-secondary btn-sm ask-cancel"></button>'
        + '<button type="button" class="btn btn-primary btn-sm ask-ok"></button>'
        + '</div></div>';
    document.body.appendChild(modal);
}

function toast(message, type = 'info', timeout = 5000) {
    _notifyInit();
    const stack = document.getElementById('toast-stack');
    const box = document.createElement('div');
    box.className = `toast ${type}`;
    const text = document.createElement('div');
    text.style.flex = '1';
    text.textContent = String(message);
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'toast-close';
    close.innerText = '×';
    close.onclick = () => box.remove();
    box.append(text, close);
    stack.appendChild(box);
    if (timeout > 0) setTimeout(() => box.remove(), timeout);
    return box;
}

function askConfirm(message, options = {}) {
    _notifyInit();
    const modal = document.getElementById('ask-modal');
    const okLabel = options.okLabel || (typeof t === 'function' ? t('Continue') : 'Continue');
    const cancelLabel = options.cancelLabel || (typeof t === 'function' ? t('Cancel') : 'Cancel');
    modal.querySelector('.ask-text').textContent = String(message);
    const ok = modal.querySelector('.ask-ok');
    const cancel = modal.querySelector('.ask-cancel');
    ok.innerText = okLabel;
    cancel.innerText = cancelLabel;

    return new Promise(resolve => {
        const finish = value => {
            modal.classList.remove('active');
            ok.onclick = cancel.onclick = modal.onclick = null;
            document.removeEventListener('keydown', onKey);
            resolve(value);
        };
        const onKey = e => { if (e.key === 'Escape') finish(false); };
        ok.onclick = () => finish(true);
        cancel.onclick = () => finish(false);
        modal.onclick = e => { if (e.target === modal) finish(false); };
        document.addEventListener('keydown', onKey);
        modal.classList.add('active');
        cancel.focus();
    });
}
