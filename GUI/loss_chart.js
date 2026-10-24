// loss_chart.js — Gráfica del entrenamiento, común a todos los entrenadores.
// Training chart shared by every trainer.
//
// Se coloca sola debajo de la consola (#terminal). Cerrada por defecto: cerrada no pide nada al servidor
// ni dibuja, así que no gasta nada. Abierta lee /api/metrics (scripts/training_metrics.py): el loss de
// cada paso, el lr y, como marcadores, los previews y checkpoints de la carpeta del proyecto.
// Se redibuja solo cuando llegan datos o el usuario interactúa; el punto que late es CSS, no un bucle.
(function () {
    const POLL_FAST = 2500, POLL_SLOW = 10000;
    const COLORS = {
        raw: 'rgba(125, 211, 252, 0.28)', lossA: '#22d3ee', lossB: '#a78bfa', lr: '#f59e0b',
        ckpt: '#10b981', preview: '#f472b6', best: '#fde047', grid: 'rgba(148, 163, 184, 0.10)',
        axis: '#64748b', text: '#94a3b8'
    };
    // Cámara del 3D: giro horizontal, inclinación (desde arriba) y zoom.
    const CAM_DEFAULT = { yaw: -0.55, pitch: 0.55, zoom: 1 };
    // Colores de la superficie según la altura (loss bajo → alto).
    const HEIGHT_STOPS = [[34, 211, 238], [99, 102, 241], [244, 114, 182]];
    const tr = (k, v) => (typeof t === 'function' ? t(k, v) : k.replace(/\{(\w+)\}/g, (m, n) => (v && n in v ? v[n] : m)));

    function store(key, value) {
        try {
            if (value === undefined) return JSON.parse(localStorage.getItem('lossChart.' + key));
            localStorage.setItem('lossChart.' + key, JSON.stringify(value));
        } catch (e) { return null; }
    }

    const CSS = `
    .lc-wrap { background: #0b1222; border: 1px solid var(--panel-border, #334155); border-radius: 10px; overflow: hidden; }
    .lc-head { display: flex; align-items: center; gap: 10px; padding: 10px 14px; cursor: pointer; user-select: none; flex-wrap: wrap; }
    .lc-head:hover { background: rgba(56, 189, 248, 0.05); }
    .lc-title { font-weight: 700; font-size: 0.95rem; color: #e2e8f0; display: flex; align-items: center; gap: 8px; }
    .lc-title .lc-caret { color: #38bdf8; transition: transform .2s; display: inline-block; }
    .lc-wrap:not(.lc-collapsed) .lc-caret { transform: rotate(90deg); }
    .lc-off { font-size: 0.72rem; color: #64748b; font-weight: 500; }
    .lc-wrap:not(.lc-collapsed) .lc-off { display: none; }
    .lc-chips { display: flex; gap: 6px; flex-wrap: wrap; margin-left: auto; }
    .lc-collapsed .lc-chips { display: none; }
    .lc-chip { background: #111a2e; border: 1px solid #1e2a44; border-radius: 7px; padding: 3px 9px; font-size: 0.72rem; color: #94a3b8; white-space: nowrap; }
    .lc-chip b { color: #e2e8f0; font-weight: 700; margin-left: 4px; font-variant-numeric: tabular-nums; }
    .lc-chip.lc-down b { color: #34d399; } .lc-chip.lc-up b { color: #f87171; }
    .lc-collapsed .lc-body { display: none; }
    .lc-body { padding: 0 14px 12px; }
    .lc-tools { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; font-size: 0.76rem; color: #94a3b8; padding-bottom: 8px; }
    .lc-tools label { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; white-space: nowrap; }
    .lc-tools input[type=range] { width: 110px; accent-color: #22d3ee; }
    .lc-tools input[type=checkbox] { accent-color: #22d3ee; }
    .lc-tools .lc-smooth-val { color: #e2e8f0; font-variant-numeric: tabular-nums; min-width: 2.2em; }
    .lc-tools .lc-reset { margin-left: auto; background: #1e293b; color: #cbd5e1; border: 1px solid #334155; border-radius: 6px; padding: 3px 10px; font-size: 0.72rem; cursor: pointer; }
    .lc-tools .lc-reset:hover { border-color: #38bdf8; color: #fff; }
    .lc-mode { display: inline-flex; border: 1px solid #334155; border-radius: 7px; overflow: hidden; }
    .lc-mode button { background: #0f172a; color: #94a3b8; border: none; padding: 3px 11px; font-size: 0.72rem; font-weight: 700; cursor: pointer; }
    .lc-mode button.lc-on { background: linear-gradient(90deg, #0e7490, #6d28d9); color: #fff; }
    .lc-hint3d { font-size: 0.7rem; color: #64748b; }
    .lc-wrap:not(.lc-3d) .lc-3d-only, .lc-3d .lc-2d-only { display: none !important; }
    .lc-3d .lc-stage { height: 400px; }
    .lc-3d .lc-stage canvas { cursor: grab; }
    .lc-stage { position: relative; height: 300px; border-radius: 8px; overflow: hidden;
        background: radial-gradient(120% 90% at 50% 0%, #0f1b33 0%, #070d1a 70%); border: 1px solid #16213a; }
    .lc-stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; cursor: crosshair; touch-action: none; }
    .lc-empty { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; text-align: center;
        color: #475569; font-size: 0.85rem; padding: 20px; pointer-events: none; }
    .lc-pulse { position: absolute; width: 10px; height: 10px; margin: -5px 0 0 -5px; border-radius: 50%; background: #a78bfa;
        box-shadow: 0 0 10px #a78bfa; pointer-events: none; display: none; }
    .lc-pulse::after { content: ""; position: absolute; inset: -6px; border-radius: 50%; border: 2px solid #a78bfa; animation: lc-ping 1.6s ease-out infinite; }
    @keyframes lc-ping { from { transform: scale(.4); opacity: .9; } to { transform: scale(1.6); opacity: 0; } }
    .lc-tip { position: absolute; pointer-events: none; background: rgba(8, 13, 26, 0.94); border: 1px solid #22304d; border-radius: 8px;
        padding: 8px 10px; font-size: 0.74rem; color: #cbd5e1; display: none; z-index: 2; min-width: 150px; backdrop-filter: blur(4px);
        box-shadow: 0 8px 24px rgba(0,0,0,.45); }
    .lc-tip .lc-tip-h { color: #fff; font-weight: 700; margin-bottom: 4px; }
    .lc-tip .lc-row { display: flex; justify-content: space-between; gap: 14px; font-variant-numeric: tabular-nums; }
    .lc-tip .lc-sw { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 6px; }
    .lc-tip img { display: block; width: 180px; height: 150px; object-fit: contain; border-radius: 6px; margin-top: 6px; background: #020617; }
    .lc-tip .lc-hint { color: #64748b; margin-top: 4px; font-size: 0.68rem; }
    .lc-legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: 0.72rem; color: #94a3b8; padding-top: 8px; }
    .lc-legend span { display: inline-flex; align-items: center; gap: 6px; }
    .lc-legend i { display: inline-block; width: 16px; height: 3px; border-radius: 2px; }
    .lc-lightbox { position: fixed; inset: 0; background: rgba(2, 6, 23, 0.86); display: none; align-items: center; justify-content: center;
        z-index: 9999; flex-direction: column; gap: 10px; cursor: zoom-out; }
    .lc-lightbox img { max-width: 92vw; max-height: 84vh; border-radius: 8px; box-shadow: 0 10px 40px rgba(0,0,0,.6); }
    .lc-lightbox audio { width: min(520px, 90vw); cursor: default; }
    .lc-lightbox .lc-cap { color: #e2e8f0; font-size: 0.85rem; }
    `;

    function build() {
        const term = document.getElementById('terminal');
        if (!term || document.getElementById('loss-chart')) return;
        const style = document.createElement('style');
        style.textContent = CSS;
        document.head.appendChild(style);

        const wrap = document.createElement('div');
        wrap.id = 'loss-chart';
        wrap.className = 'lc-wrap lc-collapsed';
        wrap.innerHTML = `
            <div class="lc-head" title="${tr('Closed, the chart does not ask the server for anything or draw anything.')}">
                <span class="lc-title"><span class="lc-caret">▶</span><span>${tr('Training Chart')}</span></span>
                <span class="lc-off">${tr('off · open it to follow the loss')}</span>
                <div class="lc-chips">
                    <span class="lc-chip" data-k="step">${tr('Step')}<b class="no-i18n">–</b></span>
                    <span class="lc-chip" data-k="loss">${tr('Loss')}<b class="no-i18n">–</b></span>
                    <span class="lc-chip" data-k="best">${tr('Best')}<b class="no-i18n">–</b></span>
                    <span class="lc-chip" data-k="trend">${tr('Trend')}<b class="no-i18n">–</b></span>
                    <span class="lc-chip no-i18n" data-k="lr">lr<b>–</b></span>
                </div>
            </div>
            <div class="lc-body">
                <div class="lc-tools">
                    <span class="lc-mode no-i18n"><button type="button" data-mode="2d">2D</button><button type="button" data-mode="3d">3D</button></span>
                    <label title="${tr('How much the loss curve is smoothed. The thin line underneath is the loss of each step.')}">
                        ${tr('Smoothing')} <input type="range" min="0" max="0.99" step="0.01" data-o="smooth"><span class="lc-smooth-val no-i18n"></span></label>
                    <label class="lc-2d-only"><input type="checkbox" data-o="raw"> ${tr('Each step')}</label>
                    <label class="lc-2d-only"><input type="checkbox" data-o="lr"> ${tr('Learning rate')}</label>
                    <label title="${tr('Leaves the loss spikes out of the vertical scale so the trend is easier to see.')}"><input type="checkbox" data-o="clip"> ${tr('Clip outliers')}</label>
                    <label class="lc-2d-only"><input type="checkbox" data-o="log"> ${tr('Log scale')}</label>
                    <span class="lc-hint3d lc-3d-only">${tr('Drag to rotate, mouse wheel zooms.')}</span>
                    <button type="button" class="lc-reset lc-2d-only" title="${tr('Mouse wheel zooms, dragging moves, double click resets.')}">${tr('Reset zoom')}</button>
                    <button type="button" class="lc-reset lc-3d-only" data-reset3d title="${tr('Double click on the chart also resets it.')}">${tr('Reset view')}</button>
                </div>
                <div class="lc-stage">
                    <canvas></canvas>
                    <div class="lc-pulse"></div>
                    <div class="lc-tip no-i18n"></div>
                    <div class="lc-empty">${tr('No data yet. The chart fills in while the LoRA trains.')}</div>
                </div>
                <div class="lc-legend">
                    <span class="lc-2d-only"><i style="background:linear-gradient(90deg,${COLORS.lossA},${COLORS.lossB})"></i>${tr('smoothed loss')}</span>
                    <span class="lc-2d-only"><i style="background:${COLORS.raw}"></i>${tr('loss at each step')}</span>
                    <span class="lc-2d-only"><i style="background:${COLORS.lr}"></i>${tr('learning rate (right-hand scale)')}</span>
                    <span class="lc-3d-only"><i style="background:linear-gradient(90deg,#22d3ee,#6366f1,#f472b6)"></i>${tr('height and colour: loss (low to high)')}</span>
                    <span class="lc-3d-only">${tr('front edge: loss at each step · back edge: smoothed loss')}</span>
                    <span><i style="background:${COLORS.preview};width:8px;height:8px;border-radius:50%"></i>${tr('preview (hover to see it)')}</span>
                    <span><i style="background:${COLORS.ckpt};width:8px;height:8px;transform:rotate(45deg)"></i>${tr('checkpoint saved')}</span>
                    <span><i style="background:${COLORS.best};width:8px;height:8px;border-radius:50%"></i>${tr('lowest smoothed loss')}</span>
                </div>
            </div>`;
        term.insertAdjacentElement('afterend', wrap);

        const lb = document.createElement('div');
        lb.className = 'lc-lightbox no-i18n';
        const closeLb = () => { lb.style.display = 'none'; lb.innerHTML = ''; };
        lb.addEventListener('click', e => { if (e.target === lb) closeLb(); });
        document.addEventListener('keydown', e => { if (e.key === 'Escape' && lb.style.display === 'flex') closeLb(); });
        document.body.appendChild(lb);

        new LossChart(wrap, lb);
    }

    class LossChart {
        constructor(wrap, lightbox) {
            this.wrap = wrap;
            this.lightbox = lightbox;
            this.canvas = wrap.querySelector('canvas');
            this.ctx = this.canvas.getContext('2d');
            this.tip = wrap.querySelector('.lc-tip');
            this.pulse = wrap.querySelector('.lc-pulse');
            this.empty = wrap.querySelector('.lc-empty');
            this.opts = Object.assign({ smooth: 0.9, raw: true, lr: true, clip: true, log: false, mode: '2d' }, store('opts') || {});
            this.cam = Object.assign({}, CAM_DEFAULT);
            this.surf = null;          // superficie 3D calculada; se rehace al llegar datos o cambiar opciones
            this.points = []; this.previews = []; this.checkpoints = []; this.run = null;
            this.view = null;          // [x0, x1] con zoom; null = todo
            this.hover = null;         // posición del ratón sobre el canvas
            this.timer = null; this.idle = 0; this.busy = false;
            this.imgCache = new Map();

            wrap.querySelector('.lc-head').addEventListener('click', () => this.setOpen(wrap.classList.contains('lc-collapsed')));
            wrap.querySelectorAll('[data-o]').forEach(el => {
                const k = el.dataset.o;
                if (el.type === 'checkbox') el.checked = !!this.opts[k]; else el.value = this.opts[k];
                el.addEventListener('input', () => {
                    this.opts[k] = el.type === 'checkbox' ? el.checked : parseFloat(el.value);
                    store('opts', this.opts);
                    this.recompute(); this.draw();
                });
            });
            wrap.querySelector('.lc-reset').addEventListener('click', () => { this.view = null; this.draw(); });
            wrap.querySelector('[data-reset3d]').addEventListener('click', () => { this.cam = Object.assign({}, CAM_DEFAULT); this.draw(); });
            wrap.querySelectorAll('.lc-mode button').forEach(b => b.addEventListener('click', () => this.setMode(b.dataset.mode)));
            this.setMode(this.opts.mode);
            this.bindPointer();
            new ResizeObserver(() => this.draw()).observe(this.canvas);
            document.addEventListener('visibilitychange', () => { if (!document.hidden && this.isOpen()) this.poll(); });
            this.setOpen(store('open') === true);
        }

        isOpen() { return !this.wrap.classList.contains('lc-collapsed'); }

        is3d() { return this.opts.mode === '3d'; }

        setMode(mode) {
            this.opts.mode = mode === '3d' ? '3d' : '2d';
            store('opts', this.opts);
            this.wrap.classList.toggle('lc-3d', this.is3d());
            this.wrap.querySelectorAll('.lc-mode button').forEach(b => b.classList.toggle('lc-on', b.dataset.mode === this.opts.mode));
            this.hover = null; this.tip.style.display = 'none';
            this.draw();
        }

        setOpen(open) {
            this.wrap.classList.toggle('lc-collapsed', !open);
            store('open', open);
            clearTimeout(this.timer);
            if (open) { this.idle = 0; this.poll(); this.draw(); }
            else this.pulse.style.display = 'none';
        }

        // ------------------------------------------------------------------ datos
        async poll() {
            clearTimeout(this.timer);
            if (!this.isOpen()) return;
            if (document.hidden) return;   // al volver a la pestaña se reanuda (visibilitychange)
            if (!this.busy) {
                this.busy = true;
                try {
                    const since = this.points.length ? this.points[this.points.length - 1].s : 0;
                    let data = await (await fetch(`/api/metrics?since=${since}`, { cache: 'no-store' })).json();
                    if (data.run !== this.run && since > 0) {
                        // Entrenamiento nuevo o reanudado: se pide el historial entero.
                        data = await (await fetch('/api/metrics?since=0', { cache: 'no-store' })).json();
                        this.points = []; this.view = null;
                    }
                    this.run = data.run;
                    const grew = data.points.length > 0;
                    if (grew) this.points.push(...data.points);
                    const markersChanged = JSON.stringify([data.previews, data.checkpoints]) !== JSON.stringify([this.previews, this.checkpoints]);
                    this.previews = data.previews; this.checkpoints = data.checkpoints;
                    this.idle = grew ? 0 : this.idle + 1;
                    if (grew || markersChanged || !this.points.length) { this.recompute(); this.draw(); }
                } catch (e) {
                    this.idle++;
                }
                this.busy = false;
            }
            // Sin pasos nuevos (entrenamiento parado) se pregunta mucho menos.
            this.timer = setTimeout(() => this.poll(), this.idle > 3 ? POLL_SLOW : POLL_FAST);
        }

        recompute() {
            this.surf = null;
            const w = Math.min(0.99, Math.max(0, this.opts.smooth));
            let last = 0, n = 0, best = null;
            for (const p of this.points) {
                // EMA con corrección de sesgo (como TensorBoard): no arranca pegada a cero.
                last = last * w + (1 - w) * p.l; n++;
                p.sm = w > 0 ? last / (1 - Math.pow(w, n)) : p.l;
                if (!best || p.sm < best.sm) best = p;
            }
            this.best = best;
            this.updateChips();
        }

        updateChips() {
            const pts = this.points, set = (k, v, cls) => {
                const chip = this.wrap.querySelector(`.lc-chip[data-k="${k}"]`);
                chip.querySelector('b').textContent = v;
                chip.classList.remove('lc-up', 'lc-down');
                if (cls) chip.classList.add(cls);
            };
            if (!pts.length) { ['step', 'loss', 'best', 'trend', 'lr'].forEach(k => set(k, '–')); return; }
            const lastP = pts[pts.length - 1];
            set('step', lastP.n ? `${lastP.s}/${lastP.n}` : String(lastP.s));
            set('loss', fmt(lastP.sm));
            set('best', `${fmt(this.best.sm)} @ ${this.best.s}`);
            set('lr', lastP.lr != null ? lastP.lr.toExponential(2) : '–');
            // Tendencia: suavizado de ahora frente al de hace un 10 % del recorrido.
            const back = pts[Math.max(0, pts.length - 1 - Math.max(5, Math.round(pts.length * 0.1)))];
            if (pts.length > 10 && back.sm) {
                const d = (lastP.sm - back.sm) / back.sm * 100;
                set('trend', `${d > 0 ? '▲' : '▼'} ${Math.abs(d).toFixed(1)}%`, d > 0 ? 'lc-up' : 'lc-down');
            } else set('trend', '–');
        }

        // ---------------------------------------------------------------- escalas
        layout() {
            const r = this.canvas.getBoundingClientRect();
            const pad = { l: 56, r: this.opts.lr ? 64 : 18, t: 16, b: 30 };
            return { W: r.width, H: r.height, pad, pw: Math.max(10, r.width - pad.l - pad.r), ph: Math.max(10, r.height - pad.t - pad.b) };
        }

        domain() {
            const pts = this.points;
            const sMin = pts.length ? pts[0].s : 0;
            const total = pts.length ? (pts[pts.length - 1].n || pts[pts.length - 1].s) : 1;
            const full = [Math.max(0, sMin - 1), Math.max(sMin + 1, total)];
            const [x0, x1] = this.view || full;
            const vis = pts.filter(p => p.s >= x0 && p.s <= x1);
            const ys = vis.map(p => this.opts.raw ? p.l : p.sm).concat(vis.map(p => p.sm)).filter(v => v > 0 || !this.opts.log);
            let y0 = Infinity, y1 = -Infinity;
            if (ys.length) {
                if (this.opts.clip && ys.length > 20) {
                    const s = ys.slice().sort((a, b) => a - b);
                    y0 = s[Math.floor(s.length * 0.01)]; y1 = s[Math.ceil(s.length * 0.98) - 1];
                    for (const p of vis) { y0 = Math.min(y0, p.sm); y1 = Math.max(y1, p.sm); }
                } else { y0 = Math.min(...ys); y1 = Math.max(...ys); }
            }
            if (!isFinite(y0)) { y0 = 0; y1 = 1; }
            if (y1 - y0 < 1e-9) { y0 -= 0.05 * Math.abs(y0 || 1); y1 += 0.05 * Math.abs(y1 || 1); }
            const m = (y1 - y0) * 0.08;
            if (this.opts.log) { y0 = Math.max(y0 * 0.95, 1e-8); y1 *= 1.05; } else { y0 -= m; y1 += m; }
            const lrs = vis.map(p => p.lr).filter(v => v != null);
            const lrMax = lrs.length ? Math.max(...lrs) * 1.1 : 1;
            return { x0, x1, y0, y1, lrMax, vis, full };
        }

        scales(L, D) {
            const sx = s => L.pad.l + (s - D.x0) / (D.x1 - D.x0 || 1) * L.pw;
            const ly = v => Math.log10(Math.max(v, 1e-12));
            const sy = this.opts.log
                ? v => L.pad.t + (1 - (ly(v) - ly(D.y0)) / (ly(D.y1) - ly(D.y0) || 1)) * L.ph
                : v => L.pad.t + (1 - (v - D.y0) / (D.y1 - D.y0 || 1)) * L.ph;
            const slr = v => L.pad.t + (1 - v / (D.lrMax || 1)) * L.ph;
            const inv = x => D.x0 + (x - L.pad.l) / L.pw * (D.x1 - D.x0);
            return { sx, sy, slr, inv };
        }

        // ---------------------------------------------------------------- dibujo
        draw() {
            if (!this.isOpen()) return;
            if (this.raf) return;
            this.raf = requestAnimationFrame(() => { this.raf = null; this.render(); });
        }

        render() {
            const L = this.layout(), dpr = window.devicePixelRatio || 1, ctx = this.ctx;
            if (L.W < 20) return;
            this.canvas.width = Math.round(L.W * dpr); this.canvas.height = Math.round(L.H * dpr);
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, L.W, L.H);
            this.empty.style.display = this.points.length ? 'none' : 'flex';
            if (!this.points.length) { this.pulse.style.display = 'none'; this.tip.style.display = 'none'; return; }
            if (this.is3d()) { this.pulse.style.display = 'none'; this.render3d(L); return; }

            const D = this.domain(), S = this.scales(L, D);
            this.L = L; this.D = D; this.S = S;
            ctx.font = '11px Segoe UI, sans-serif';

            // Rejilla y ejes
            ctx.strokeStyle = COLORS.grid; ctx.lineWidth = 1; ctx.fillStyle = COLORS.text;
            ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
            for (const v of yTicks(D.y0, D.y1, this.opts.log)) {
                const y = Math.round(S.sy(v)) + 0.5;
                if (y < L.pad.t - 1 || y > L.pad.t + L.ph + 1) continue;
                ctx.beginPath(); ctx.moveTo(L.pad.l, y); ctx.lineTo(L.pad.l + L.pw, y); ctx.stroke();
                ctx.fillText(fmt(v), L.pad.l - 8, y);
            }
            ctx.textAlign = 'center'; ctx.textBaseline = 'top';
            for (const s of niceTicks(D.x0, D.x1, Math.max(2, Math.floor(L.pw / 90)))) {
                const x = Math.round(S.sx(s)) + 0.5;
                ctx.beginPath(); ctx.moveTo(x, L.pad.t); ctx.lineTo(x, L.pad.t + L.ph); ctx.stroke();
                ctx.fillText(String(Math.round(s)), x, L.pad.t + L.ph + 8);
            }
            if (this.opts.lr) {
                ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = 'rgba(245, 158, 11, 0.75)';
                for (let i = 0; i <= 4; i++) {
                    const v = D.lrMax * i / 4;
                    ctx.fillText(v.toExponential(1), L.pad.l + L.pw + 8, S.slr(v));
                }
            }

            ctx.save();
            ctx.beginPath(); ctx.rect(L.pad.l, L.pad.t - 2, L.pw, L.ph + 4); ctx.clip();

            // Checkpoints: línea vertical tenue
            ctx.setLineDash([3, 4]); ctx.strokeStyle = 'rgba(16, 185, 129, 0.28)';
            for (const c of this.checkpoints) {
                if (c < D.x0 || c > D.x1) continue;
                const x = Math.round(S.sx(c)) + 0.5;
                ctx.beginPath(); ctx.moveTo(x, L.pad.t); ctx.lineTo(x, L.pad.t + L.ph); ctx.stroke();
            }
            ctx.setLineDash([]);

            const vis = decimate(D.vis, L.pw * 2);
            // Loss de cada paso
            if (this.opts.raw) {
                ctx.strokeStyle = COLORS.raw; ctx.lineWidth = 1;
                ctx.beginPath();
                vis.forEach((p, i) => { const x = S.sx(p.s), y = S.sy(p.l); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
                ctx.stroke();
            }

            // Learning rate
            if (this.opts.lr) {
                const lrPts = vis.filter(p => p.lr != null);
                if (lrPts.length) {
                    ctx.strokeStyle = COLORS.lr; ctx.lineWidth = 1.6; ctx.globalAlpha = 0.85;
                    ctx.beginPath();
                    lrPts.forEach((p, i) => { const x = S.sx(p.s), y = S.slr(p.lr); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
                    ctx.stroke(); ctx.globalAlpha = 1;
                }
            }

            // Loss suavizado: relleno degradado + línea con brillo
            if (vis.length) {
                const grad = ctx.createLinearGradient(L.pad.l, 0, L.pad.l + L.pw, 0);
                grad.addColorStop(0, COLORS.lossA); grad.addColorStop(1, COLORS.lossB);
                const path = new Path2D();
                vis.forEach((p, i) => { const x = S.sx(p.s), y = S.sy(p.sm); i ? path.lineTo(x, y) : path.moveTo(x, y); });
                const area = new Path2D(path);
                area.lineTo(S.sx(vis[vis.length - 1].s), L.pad.t + L.ph);
                area.lineTo(S.sx(vis[0].s), L.pad.t + L.ph);
                area.closePath();
                const fill = ctx.createLinearGradient(0, L.pad.t, 0, L.pad.t + L.ph);
                fill.addColorStop(0, 'rgba(56, 189, 248, 0.22)'); fill.addColorStop(1, 'rgba(139, 92, 246, 0.0)');
                ctx.fillStyle = fill; ctx.fill(area);
                ctx.strokeStyle = grad; ctx.lineWidth = 2.6; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
                ctx.shadowColor = 'rgba(56, 189, 248, 0.65)'; ctx.shadowBlur = 12;
                ctx.stroke(path);
                ctx.shadowBlur = 0;
            }

            // Mejor punto
            if (this.best && this.best.s >= D.x0 && this.best.s <= D.x1) {
                const x = S.sx(this.best.s), y = S.sy(this.best.sm);
                ctx.fillStyle = COLORS.best; ctx.shadowColor = COLORS.best; ctx.shadowBlur = 10;
                ctx.beginPath(); ctx.arc(x, y, 4.5, 0, Math.PI * 2); ctx.fill(); ctx.shadowBlur = 0;
            }
            ctx.restore();

            // Marcadores sobre el eje y la curva
            this.hits = [];
            for (const c of this.checkpoints) {
                if (c < D.x0 || c > D.x1) continue;
                const x = S.sx(c), y = L.pad.t + L.ph;
                ctx.fillStyle = COLORS.ckpt;
                ctx.beginPath(); ctx.moveTo(x, y - 5); ctx.lineTo(x + 5, y); ctx.lineTo(x, y + 5); ctx.lineTo(x - 5, y); ctx.closePath(); ctx.fill();
                this.hits.push({ kind: 'ckpt', step: c, x, y });
            }
            for (const [step, file] of this.previews) {
                if (step < D.x0 || step > D.x1) continue;
                const p = nearest(this.points, step);
                if (!p) continue;
                const x = S.sx(step), y = Math.min(L.pad.t + L.ph, Math.max(L.pad.t, S.sy(p.sm)));
                ctx.fillStyle = '#0b1222'; ctx.strokeStyle = COLORS.preview; ctx.lineWidth = 2;
                ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
                this.hits.push({ kind: 'preview', step, file, x, y });
            }

            // Punto vivo al final de la curva (la animación es CSS)
            const lastP = this.points[this.points.length - 1];
            const live = this.idle <= 3 && lastP.s >= D.x0 && lastP.s <= D.x1;
            if (live) {
                this.pulse.style.display = 'block';
                this.pulse.style.left = S.sx(lastP.s) + 'px';
                this.pulse.style.top = Math.min(L.pad.t + L.ph, Math.max(L.pad.t, S.sy(lastP.sm))) + 'px';
            } else this.pulse.style.display = 'none';

            this.drawHover();
        }

        // ------------------------------------------------------------------- 3D
        // Superficie: ancho = pasos, alto = loss, fondo = suavizado. La fila delantera es el loss de cada
        // paso y cada fila hacia atrás está más suavizada, hasta el suavizado elegido en el control.
        build3d() {
            const pts = this.points, n = pts.length;
            const C = Math.max(2, Math.min(100, n)), R = 20;
            const wmax = Math.min(0.99, Math.max(0.6, this.opts.smooth));
            const cols = Array.from({ length: C }, (_, c) => Math.round(c * (n - 1) / (C - 1)));
            const rows = [];
            for (let r = 0; r < R; r++) {
                const w = r === 0 ? 0 : 1 - Math.pow(1 - wmax, r / (R - 1));
                const vals = new Array(n);
                let last = 0;
                for (let i = 0; i < n; i++) {
                    last = last * w + (1 - w) * pts[i].l;
                    vals[i] = w > 0 ? last / (1 - Math.pow(w, i + 1)) : pts[i].l;
                }
                // La fila delantera es la media de cada tramo: sigue siendo el ruido de los pasos, sin pinchos sueltos.
                rows.push(r > 0 ? cols.map(i => vals[i]) : cols.map((i, c) => {
                    const a = c ? Math.floor((cols[c - 1] + i) / 2) + 1 : 0, b = c < C - 1 ? Math.floor((i + cols[c + 1]) / 2) : n - 1;
                    let sum = 0; for (let k = a; k <= b; k++) sum += vals[k];
                    return sum / (b - a + 1);
                }));
            }
            const all = rows.flat().sort((a, b) => a - b);
            let y0 = all[0], y1 = all[all.length - 1];
            if (this.opts.clip && all.length > 50) { y0 = all[Math.floor(all.length * 0.01)]; y1 = all[Math.ceil(all.length * 0.99) - 1]; }
            if (y1 - y0 < 1e-9) { y0 -= 0.05; y1 += 0.05; }
            const pad = (y1 - y0) * 0.05; y0 -= pad; y1 += pad;
            const H = 0.75, Z = 0.55;
            const world = rows.map((row, r) => row.map((v, c) => [
                -1 + 2 * c / (C - 1),
                Math.min(1, Math.max(0, (v - y0) / (y1 - y0))) * H,
                -Z + 2 * Z * r / (R - 1)
            ]));
            const steps = cols.map(i => pts[i].s);
            this.surf = { R, C, rows, world, steps, y0, y1, H, Z };
            return this.surf;
        }

        project(x, y, z, L) {
            const { yaw, pitch, zoom } = this.cam;
            y -= 0.32;
            const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
            const x1 = x * cy - z * sy, z1 = x * sy + z * cy;
            const y2 = y * cp + z1 * sp, z2 = z1 * cp - y * sp;
            const dist = 3.4, d = dist + z2;
            const f = Math.min(L.W * 0.36, L.H * 0.62) * zoom * dist / d;
            return [L.W * 0.54 + x1 * f, L.H * 0.52 - y2 * f, d];
        }

        render3d(L) {
            const ctx = this.ctx, S3 = this.surf || this.build3d(), P = (x, y, z) => this.project(x, y, z, L);
            const { R, C, world, steps, y0, y1, H, Z } = S3;
            this.L = L; this.D = null;
            ctx.font = '11px Segoe UI, sans-serif';
            ctx.lineJoin = 'round';

            // Suelo y pared del fondo con la escala del loss
            ctx.strokeStyle = COLORS.grid; ctx.lineWidth = 1; ctx.fillStyle = COLORS.text;
            const s0 = steps[0], s1 = steps[C - 1], xOf = s => -1 + 2 * (s - s0) / ((s1 - s0) || 1);
            const line = (a, b) => { const p = P(...a), q = P(...b); ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke(); };
            ctx.fillStyle = 'rgba(15, 23, 42, 0.55)';
            const fl = [P(-1, 0, -Z), P(1, 0, -Z), P(1, 0, Z), P(-1, 0, Z)];
            ctx.beginPath(); fl.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.closePath(); ctx.fill();
            ctx.fillStyle = COLORS.text;
            const xt = niceTicks(s0, s1, 5);
            for (const s of xt) {
                const x = xOf(s);
                line([x, 0, -Z], [x, 0, Z]);
                const p = P(x, 0, -Z - 0.09);
                ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(String(Math.round(s)), p[0], p[1]);
            }
            for (let k = 0; k <= 4; k++) line([-1, 0, -Z + 2 * Z * k / 4], [1, 0, -Z + 2 * Z * k / 4]);
            const yTicks3 = niceTicks(y0, y1, 4).map(v => [v, (v - y0) / (y1 - y0) * H]).filter(([, y]) => y >= 0 && y <= H);
            for (const [, y] of yTicks3) line([-1, y, Z], [1, y, Z]);
            line([-1, 0, Z], [-1, H, Z]); line([1, 0, Z], [1, H, Z]);

            // Cuadriláteros ordenados de lejos a cerca (pintor), con color por altura y luz
            const pr = world.map(row => row.map(w => P(w[0], w[1], w[2])));
            const quads = [];
            for (let r = 0; r < R - 1; r++) for (let c = 0; c < C - 1; c++) {
                const a = pr[r][c], b = pr[r][c + 1], d = pr[r + 1][c + 1], e = pr[r + 1][c];
                quads.push([(a[2] + b[2] + d[2] + e[2]) / 4, r, c]);
            }
            quads.sort((u, v) => v[0] - u[0]);
            const light = norm3([-0.35, 0.85, -0.4]);
            for (const [, r, c] of quads) {
                const wa = world[r][c], wb = world[r][c + 1], we = world[r + 1][c];
                const nrm = norm3(cross3(sub3(we, wa), sub3(wb, wa)));
                const shade = 0.5 + 0.5 * Math.abs(dot3(nrm, light));
                const h = (wa[1] + wb[1] + world[r + 1][c + 1][1] + we[1]) / (4 * H);
                const col = heightColor(h, shade);
                const a = pr[r][c], b = pr[r][c + 1], d = pr[r + 1][c + 1], e = pr[r + 1][c];
                ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.lineTo(d[0], d[1]); ctx.lineTo(e[0], e[1]); ctx.closePath();
                ctx.fillStyle = col; ctx.strokeStyle = col; ctx.lineWidth = 0.6;
                ctx.fill(); ctx.stroke();
                // Rejilla sutil y bordes: delante el loss de cada paso, detrás la tendencia con brillo
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.07)';
                if (c % 6 === 0) { ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(e[0], e[1]); ctx.stroke(); }
                if (r % 4 === 0) { ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); }
                if (r === 0) {
                    ctx.strokeStyle = 'rgba(165, 243, 252, 0.9)'; ctx.lineWidth = 1.2;
                    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
                }
                if (r === R - 2) {
                    ctx.strokeStyle = '#c4b5fd'; ctx.lineWidth = 2.2; ctx.shadowColor = 'rgba(167, 139, 250, 0.9)'; ctx.shadowBlur = 8;
                    ctx.beginPath(); ctx.moveTo(e[0], e[1]); ctx.lineTo(d[0], d[1]); ctx.stroke();
                    ctx.shadowBlur = 0;
                }
            }

            // Paredes del bloque hasta el suelo, solo las que miran a la cámara: parece un trozo de terreno
            const { yaw, pitch } = this.cam;
            const camDir = [-Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch), -Math.cos(pitch) * Math.cos(yaw)];
            const walls = [
                { n: [0, 0, -1], top: world[0], edge: 'rgba(165, 243, 252, 0.95)' },
                { n: [0, 0, 1], top: world[R - 1], edge: '#c4b5fd' },
                { n: [-1, 0, 0], top: world.map(row => row[0]) },
                { n: [1, 0, 0], top: world.map(row => row[C - 1]) },
            ];
            for (const wall of walls) {
                if (dot3(wall.n, camDir) <= 0) continue;
                const top = wall.top.map(w => P(...w)), floor = wall.top.map(w => P(w[0], 0, w[2])).reverse();
                const ys = top.map(p => p[1]).concat(floor.map(p => p[1]));
                const g = ctx.createLinearGradient(0, Math.min(...ys), 0, Math.max(...ys));
                g.addColorStop(0, 'rgba(79, 70, 229, 0.55)'); g.addColorStop(1, 'rgba(8, 13, 30, 0.92)');
                ctx.beginPath();
                top.concat(floor).forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
                ctx.closePath(); ctx.fillStyle = g; ctx.fill();
                if (wall.edge) {
                    ctx.strokeStyle = wall.edge; ctx.lineWidth = 1.4;
                    ctx.beginPath(); top.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.stroke();
                }
            }

            // Escala del loss junto a la arista delantera izquierda, donde no tapa la superficie
            ctx.fillStyle = COLORS.text; ctx.font = '11px Segoe UI, sans-serif';
            for (const [v, y] of yTicks3) {
                const p = P(-1.06, y, -Z);
                ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(fmt(v), p[0], p[1]);
            }

            // Marcadores: previews y mejor punto sobre la tendencia, checkpoints en el suelo
            this.hits = [];
            const back = world[R - 1], colOf = s => { let best = 0; for (let c = 0; c < C; c++) if (Math.abs(steps[c] - s) < Math.abs(steps[best] - s)) best = c; return best; };
            for (const ck of this.checkpoints) {
                if (ck < s0 || ck > s1) continue;
                const p = P(xOf(ck), 0, Z + 0.05);
                ctx.fillStyle = COLORS.ckpt;
                ctx.beginPath(); ctx.moveTo(p[0], p[1] - 4); ctx.lineTo(p[0] + 4, p[1]); ctx.lineTo(p[0], p[1] + 4); ctx.lineTo(p[0] - 4, p[1]); ctx.closePath(); ctx.fill();
                this.hits.push({ kind: 'ckpt', step: ck, x: p[0], y: p[1] });
            }
            for (const [step, file] of this.previews) {
                if (step < s0 || step > s1) continue;
                const w = back[colOf(step)], top = P(xOf(step), w[1] + 0.08, Z), foot = P(xOf(step), w[1], Z);
                ctx.strokeStyle = 'rgba(244, 114, 182, 0.6)'; ctx.lineWidth = 1;
                ctx.beginPath(); ctx.moveTo(foot[0], foot[1]); ctx.lineTo(top[0], top[1]); ctx.stroke();
                ctx.fillStyle = '#0b1222'; ctx.strokeStyle = COLORS.preview; ctx.lineWidth = 2;
                ctx.beginPath(); ctx.arc(top[0], top[1], 4.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
                this.hits.push({ kind: 'preview', step, file, x: top[0], y: top[1] });
            }
            if (this.best) {
                const w = back[colOf(this.best.s)], p = P(xOf(this.best.s), w[1], Z);
                ctx.fillStyle = COLORS.best; ctx.shadowColor = COLORS.best; ctx.shadowBlur = 10;
                ctx.beginPath(); ctx.arc(p[0], p[1], 4.5, 0, Math.PI * 2); ctx.fill(); ctx.shadowBlur = 0;
            }

            // Rótulos de las dos orillas
            ctx.font = '600 11px Segoe UI, sans-serif'; ctx.textBaseline = 'middle';
            let p = P(-1, world[0][0][1] + 0.14, -Z); ctx.textAlign = 'center'; ctx.fillStyle = '#a5f3fc'; ctx.fillText(tr('each step'), p[0], p[1]);
            p = P(1, back[C - 1][1] + 0.2, Z); ctx.textAlign = 'right'; ctx.fillStyle = '#c4b5fd'; ctx.fillText(tr('smoothed'), p[0], p[1]);

            // El punto que late, al final de la tendencia mientras llegan pasos
            if (this.idle <= 3) {
                const q = P(1, back[C - 1][1], Z);
                this.pulse.style.display = 'block'; this.pulse.style.left = q[0] + 'px'; this.pulse.style.top = q[1] + 'px';
            }
            this.hover3d();
        }

        hover3d() {
            const h = this.hover;
            const hit = h && (this.hits || []).find(m => Math.hypot(m.x - h.x, m.y - h.y) < 9);
            if (!hit) { this.tip.style.display = 'none'; this.canvas.style.cursor = this.drag ? 'grabbing' : ''; return; }
            this.showTip(hit, null);
        }

        drawHover() {
            const h = this.hover, ctx = this.ctx, L = this.L, S = this.S, D = this.D;
            if (!h || !L || h.x < L.pad.l || h.x > L.pad.l + L.pw || h.y < 0 || h.y > L.H) { this.tip.style.display = 'none'; return; }
            const hit = (this.hits || []).find(m => Math.hypot(m.x - h.x, m.y - h.y) < 9);
            const p = nearest(D.vis, S.inv(h.x));
            if (!p) { this.tip.style.display = 'none'; return; }
            const x = hit ? hit.x : S.sx(p.s);
            ctx.strokeStyle = 'rgba(226, 232, 240, 0.25)'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(Math.round(x) + 0.5, L.pad.t); ctx.lineTo(Math.round(x) + 0.5, L.pad.t + L.ph); ctx.stroke();
            if (!hit) {
                ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(x, S.sy(p.sm), 3.5, 0, Math.PI * 2); ctx.fill();
            }
            this.showTip(hit, p);
        }

        // Ventanita junto al ratón: un preview, un checkpoint o (en 2D) los valores del paso p.
        showTip(hit, p) {
            const h = this.hover, L = this.L;
            let html;
            if (hit && hit.kind === 'preview') {
                const url = '/api/preview/' + encodeURIComponent(hit.file);
                const isImg = /\.(png|jpe?g|webp)$/i.test(hit.file);
                html = `<div class="lc-tip-h">${tr('Preview')} · ${tr('Step')} ${hit.step}</div>` +
                    (isImg ? `<img src="${url}" alt="">` : `<div>${escapeHtml(hit.file)}</div>`) +
                    `<div class="lc-hint">${tr('Click to open it')}</div>`;
            } else if (hit && hit.kind === 'ckpt') {
                html = `<div class="lc-tip-h">${tr('Checkpoint saved')} · ${tr('Step')} ${hit.step}</div>`;
            } else {
                const row = (c, k, v) => `<div class="lc-row"><span><i class="lc-sw" style="background:${c}"></i>${k}</span><b>${v}</b></div>`;
                html = `<div class="lc-tip-h">${tr('Step')} ${p.s}${p.n ? ' / ' + p.n : ''}</div>` +
                    row(COLORS.lossA, tr('smoothed'), fmt(p.sm)) +
                    row('rgba(125, 211, 252, 0.6)', tr('this step'), fmt(p.l)) +
                    (p.lr != null ? row(COLORS.lr, 'lr', p.lr.toExponential(2)) : '') +
                    (p.g != null ? row('#64748b', 'gnorm', p.g.toFixed(3)) : '') +
                    (p.dt != null ? row('#64748b', 's/it', p.dt.toFixed(2)) : '');
            }
            this.tip.innerHTML = html;
            this.tip.style.display = 'block';
            const tw = this.tip.offsetWidth, th = this.tip.offsetHeight;
            let left = h.x + 16, top = h.y + 12;
            if (left + tw > L.W - 4) left = h.x - tw - 16;
            if (top + th > L.H - 4) top = Math.max(4, L.H - th - 4);
            this.tip.style.left = left + 'px'; this.tip.style.top = top + 'px';
            this.canvas.style.cursor = hit && hit.kind === 'preview' ? 'pointer'
                : this.drag ? 'grabbing' : this.is3d() ? '' : 'crosshair';
        }

        // ---------------------------------------------------------- interacción
        bindPointer() {
            const c = this.canvas;
            const pos = e => { const r = c.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
            c.addEventListener('pointermove', e => {
                this.hover = pos(e);
                if (this.drag && this.drag.cam) {
                    const dx = this.hover.x - this.drag.x, dy = this.hover.y - this.drag.y;
                    this.cam.yaw = this.drag.cam.yaw + dx * 0.008;
                    this.cam.pitch = Math.min(1.45, Math.max(0.05, this.drag.cam.pitch + dy * 0.006));
                    if (Math.abs(dx) + Math.abs(dy) > 3) this.drag.moved = true;
                } else if (this.drag && this.D) {
                    const dx = this.hover.x - this.drag.x, span = this.drag.view[1] - this.drag.view[0];
                    const shift = -dx / this.L.pw * span;
                    this.view = this.clampView([this.drag.view[0] + shift, this.drag.view[1] + shift]);
                    if (Math.abs(dx) > 3) this.drag.moved = true;
                }
                this.draw();
            });
            c.addEventListener('pointerleave', () => { this.hover = null; this.draw(); });
            c.addEventListener('pointerdown', e => {
                if (this.is3d()) {
                    c.setPointerCapture(e.pointerId);
                    const p = pos(e);
                    this.drag = { x: p.x, y: p.y, cam: Object.assign({}, this.cam), moved: false };
                    return;
                }
                if (!this.D) return;
                c.setPointerCapture(e.pointerId);
                this.drag = { x: pos(e).x, view: [this.D.x0, this.D.x1], moved: false };
            });
            c.addEventListener('pointerup', e => {
                const d = this.drag; this.drag = null;
                if (d && !d.moved) this.click(pos(e));
                this.draw();
            });
            c.addEventListener('dblclick', () => {
                if (this.is3d()) this.cam = Object.assign({}, CAM_DEFAULT); else this.view = null;
                this.draw();
            });
            c.addEventListener('wheel', e => {
                if (this.is3d()) {
                    e.preventDefault();
                    this.cam.zoom = Math.min(2.6, Math.max(0.6, this.cam.zoom * (e.deltaY > 0 ? 1 / 1.12 : 1.12)));
                    this.draw();
                    return;
                }
                if (!this.D || !this.points.length) return;
                e.preventDefault();
                const at = this.S.inv(pos(e).x), f = e.deltaY > 0 ? 1.2 : 1 / 1.2;
                const x0 = at - (at - this.D.x0) * f, x1 = at + (this.D.x1 - at) * f;
                this.view = x1 - x0 < 5 ? this.view : this.clampView([x0, x1]);
                this.draw();
            }, { passive: false });
        }

        clampView([x0, x1]) {
            const [f0, f1] = this.D.full, span = Math.min(x1 - x0, f1 - f0);
            if (span >= f1 - f0) return null;
            if (x0 < f0) { x0 = f0; x1 = f0 + span; }
            if (x1 > f1) { x1 = f1; x0 = f1 - span; }
            return [x0, x1];
        }

        click(p) {
            const hit = (this.hits || []).find(m => m.kind === 'preview' && Math.hypot(m.x - p.x, m.y - p.y) < 9);
            if (!hit) return;
            const url = '/api/preview/' + encodeURIComponent(hit.file);
            const media = /\.(png|jpe?g|webp)$/i.test(hit.file) ? `<img src="${url}" alt="">`
                : /\.(wav|mp3)$/i.test(hit.file) ? `<audio controls autoplay src="${url}"></audio>`
                : `<video controls autoplay src="${url}" style="max-width:92vw;max-height:84vh"></video>`;
            this.lightbox.innerHTML = media + `<div class="lc-cap">${tr('Preview')} · ${tr('Step')} ${hit.step}</div>`;
            this.lightbox.style.display = 'flex';
        }
    }

    // ------------------------------------------------------------------ utilidades
    function fmt(v) {
        if (v == null || !isFinite(v)) return '–';
        const a = Math.abs(v);
        return a >= 100 ? v.toFixed(1) : a >= 1 ? v.toFixed(3) : a >= 0.001 ? v.toFixed(4) : v.toExponential(2);
    }

    // Vectores 3D mínimos para la luz de la superficie.
    const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const norm3 = a => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

    function heightColor(h, shade) {
        h = Math.min(1, Math.max(0, h)) * (HEIGHT_STOPS.length - 1);
        const i = Math.min(HEIGHT_STOPS.length - 2, Math.floor(h)), f = h - i, a = HEIGHT_STOPS[i], b = HEIGHT_STOPS[i + 1];
        const ch = k => Math.round((a[k] + (b[k] - a[k]) * f) * shade);
        return `rgb(${ch(0)}, ${ch(1)}, ${ch(2)})`;
    }

    function escapeHtml(s) {
        return String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    }

    function niceTicks(a, b, count) {
        const span = b - a;
        if (!(span > 0)) return [a];
        const raw = span / count, mag = Math.pow(10, Math.floor(Math.log10(raw)));
        const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || raw;
        const out = [];
        for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(v);
        return out;
    }

    function yTicks(a, b, log) {
        if (!log) return niceTicks(a, b, 5);
        const out = [], l0 = Math.floor(Math.log10(a)), l1 = Math.ceil(Math.log10(b));
        for (let e = l0; e <= l1; e++) for (const m of [1, 2, 5]) { const v = m * Math.pow(10, e); if (v >= a && v <= b) out.push(v); }
        return out.length >= 2 ? out : niceTicks(a, b, 5);
    }

    function nearest(pts, s) {
        if (!pts.length) return null;
        let lo = 0, hi = pts.length - 1;
        while (lo < hi) { const mid = (lo + hi) >> 1; if (pts[mid].s < s) lo = mid + 1; else hi = mid; }
        return lo > 0 && Math.abs(pts[lo - 1].s - s) < Math.abs(pts[lo].s - s) ? pts[lo - 1] : pts[lo];
    }

    // Con más puntos que píxeles se queda el mínimo y el máximo de cada tramo: la forma no cambia.
    function decimate(pts, max) {
        if (pts.length <= max) return pts;
        const out = [], per = pts.length / (max / 2);
        for (let i = 0; i < pts.length; i += per) {
            const chunk = pts.slice(Math.floor(i), Math.floor(i + per));
            if (!chunk.length) continue;
            let lo = chunk[0], hi = chunk[0];
            for (const p of chunk) { if (p.l < lo.l) lo = p; if (p.l > hi.l) hi = p; }
            if (lo.s <= hi.s) out.push(lo, hi); else out.push(hi, lo);
        }
        return out;
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build);
    else build();
})();
