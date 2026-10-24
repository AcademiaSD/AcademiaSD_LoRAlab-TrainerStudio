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
                    <label title="${tr('How much the loss curve is smoothed. The thin line underneath is the loss of each step.')}">
                        ${tr('Smoothing')} <input type="range" min="0" max="0.99" step="0.01" data-o="smooth"><span class="lc-smooth-val no-i18n"></span></label>
                    <label><input type="checkbox" data-o="raw"> ${tr('Each step')}</label>
                    <label><input type="checkbox" data-o="lr"> ${tr('Learning rate')}</label>
                    <label title="${tr('Leaves the loss spikes out of the vertical scale so the trend is easier to see.')}"><input type="checkbox" data-o="clip"> ${tr('Clip outliers')}</label>
                    <label><input type="checkbox" data-o="log"> ${tr('Log scale')}</label>
                    <button type="button" class="lc-reset" title="${tr('Mouse wheel zooms, dragging moves, double click resets.')}">${tr('Reset zoom')}</button>
                </div>
                <div class="lc-stage">
                    <canvas></canvas>
                    <div class="lc-pulse"></div>
                    <div class="lc-tip no-i18n"></div>
                    <div class="lc-empty">${tr('No data yet. The chart fills in while the LoRA trains.')}</div>
                </div>
                <div class="lc-legend">
                    <span><i style="background:linear-gradient(90deg,${COLORS.lossA},${COLORS.lossB})"></i>${tr('smoothed loss')}</span>
                    <span><i style="background:${COLORS.raw}"></i>${tr('loss at each step')}</span>
                    <span><i style="background:${COLORS.lr}"></i>${tr('learning rate (right-hand scale)')}</span>
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
            this.opts = Object.assign({ smooth: 0.9, raw: true, lr: true, clip: true, log: false }, store('opts') || {});
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
            this.bindPointer();
            new ResizeObserver(() => this.draw()).observe(this.canvas);
            document.addEventListener('visibilitychange', () => { if (!document.hidden && this.isOpen()) this.poll(); });
            this.setOpen(store('open') === true);
        }

        isOpen() { return !this.wrap.classList.contains('lc-collapsed'); }

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
            this.canvas.style.cursor = hit && hit.kind === 'preview' ? 'pointer' : (this.drag ? 'grabbing' : 'crosshair');
        }

        // ---------------------------------------------------------- interacción
        bindPointer() {
            const c = this.canvas;
            const pos = e => { const r = c.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
            c.addEventListener('pointermove', e => {
                this.hover = pos(e);
                if (this.drag && this.D) {
                    const dx = this.hover.x - this.drag.x, span = this.drag.view[1] - this.drag.view[0];
                    const shift = -dx / this.L.pw * span;
                    this.view = this.clampView([this.drag.view[0] + shift, this.drag.view[1] + shift]);
                    if (Math.abs(dx) > 3) this.drag.moved = true;
                }
                this.draw();
            });
            c.addEventListener('pointerleave', () => { this.hover = null; this.draw(); });
            c.addEventListener('pointerdown', e => {
                if (!this.D) return;
                c.setPointerCapture(e.pointerId);
                this.drag = { x: pos(e).x, view: [this.D.x0, this.D.x1], moved: false };
            });
            c.addEventListener('pointerup', e => {
                const d = this.drag; this.drag = null;
                if (d && !d.moved) this.click(pos(e));
                this.draw();
            });
            c.addEventListener('dblclick', () => { this.view = null; this.draw(); });
            c.addEventListener('wheel', e => {
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
