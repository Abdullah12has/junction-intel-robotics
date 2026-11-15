/* Brick inspection loop dashboard (OD-9): renders the snapshot pushed on /api/events.
 * One card per RealSense view with its own score bar, the fused verdict on top, the
 * handover and sorter status, and the live preview (live.jpg) polled about five times a
 * second. Operator buttons POST to /api/control. No external resources; text is set with
 * textContent only. */
(() => {
  'use strict';

  const STAGES = {
    NO_RUN: ['NO RUN', 'start the loop on the laptop; this page follows the current run'],
    STARTING: ['STARTING', 'run is starting'],
    IDLE: ['IDLE', 'loop idle; press Start to begin'],
    HOME: ['HOME', 'arm at home, checking it is still'],
    SCAN: ['SCAN', 'both RealSense views list every brick on the mat'],
    TRIAGE: ['TRIAGE', 'confident defect goes straight to GREEN (I-52); the rest are inspected'],
    ACT: ['ACT', 'policy moves the brick to its plate'],
    VERIFY: ['VERIFY', 'top view checks which plate the brick landed on'],
    WAIT_BRICK: ['WAIT FOR BRICK', 'place the next brick, then press Next brick'],
    S1_PICK: ['S1 PICKING', 'policy moves the brick to the inspection spot'],
    INSPECT: ['INSPECT', 'both RealSense views check the brick at the spot'],
    HANDOVER: ['HANDOVER', 'arm released to the sorter; the policy model stays loaded'],
    SORT_BLUE: ['SORTING TO BLUE PLATE', 'scripted sorter: good brick to the blue plate'],
    SORT_GREEN: ['SORTING TO GREEN PLATE', 'scripted sorter: defective or uncertain brick to the green plate'],
    SORT: ['SORTING', 'scripted sorter running'],
    S2_BLUE: ['S2 TO BLUE PLATE', 'policy sort: good brick to the blue plate'],
    S3_GREEN: ['S3 TO GREEN PLATE', 'policy sort: defective or uncertain brick to the green plate'],
    CYCLE_DONE: ['CYCLE DONE', 'cycle complete; waiting for the next brick'],
    HOLD: ['HOLD', 'loop paused on an error; arm holds its position'],
    STOPPED: ['STOPPED', 'loop stopped; arm holds its position'],
    RUN_ENDED: ['RUN ENDED', 'run finished'],
  };
  // OD-10: the reasons shown are "tape" and/or "not 2x2". The contract keys stay as emitted;
  // no_brick (UNCERTAIN, no brick found) keeps its own label.
  const REASON_LABELS = { 'tape/appearance': 'tape', tape: 'tape', not_2x2: 'not 2x2', no_brick: 'no brick' };
  const DEVICE_NAMES = { GPU: 'Intel GPU', NPU: 'Intel NPU', CPU: 'CPU' };
  const VIEW_TITLES = { close: 'Close view', top: 'Top view' };
  const HISTORY_SLOTS = 8;
  const LIVE_MAX_AGE_S = 3;
  const LIVE_GAP_MS = 150;
  const TICK_MS = 250;
  // control.json holds only the latest command, so Start and Next wait until the
  // loop has acked the previous one (STOP is never held back).
  const ACK_TIMEOUT_MS = 3000;
  // The stream is down (not OPEN) and no snapshot has arrived for this long: server offline.
  const SERVER_DOWN_MS = 5000;

  const $ = (id) => document.getElementById(id);
  const el = {
    stage: $('stage'),
    stageSub: $('stage-sub'),
    stageTime: $('stage-time'),
    detail: $('detail'),
    runId: $('run-id'),
    modeBadge: $('mode-badge'),
    linkBadge: $('link-badge'),
    alert: $('alert'),
    live: $('live'),
    liveSource: $('live-source'),
    verdict: $('verdict'),
    verdictPlate: $('verdict-plate'),
    verdictSafe: $('verdict-safe'),
    verdictSub: $('verdict-sub'),
    inspMeta: $('insp-meta'),
    cams: $('cams'),
    reasons: $('reasons'),
    device: $('device'),
    geometry: $('geometry'),
    cCycles: $('c-cycles'),
    cGood: $('c-good'),
    cDefective: $('c-defective'),
    cUncertain: $('c-uncertain'),
    cMean: $('c-mean'),
    sSuccess: $('s-success'),
    sErrors: $('s-errors'),
    sRoute: $('s-route'),
    frameCard: $('frame-card'),
    frameImg: $('frame-img'),
    bricks: $('bricks'),
    scanMeta: $('scan-meta'),
    policyMeta: $('policy-meta'),
    decisionPlate: $('decision-plate'),
    decisionRule: $('decision-rule'),
    timeline: $('timeline'),
    verify: $('verify'),
    verifyFrame: $('verify-frame'),
    verifyImg: $('verify-img'),
    intelMeta: $('intel-meta'),
    intelLive: $('intel-live'),
    intelLat: $('intel-lat'),
    bench: $('bench'),
    intelSrc: $('intel-src'),
    btnVideo: $('btn-video'),
    btnStart: $('btn-start'),
    btnNext: $('btn-next'),
    btnStop: $('btn-stop'),
    cmdState: $('cmd-state'),
    strip: $('strip'),
  };

  let snap = null;
  let historySig = null;
  let cmdError = '';
  let cmdBusy = false;
  let pendingId = null;
  let pendingAt = 0;
  let lastSnapshotMs = 0;
  const pageStartMs = Date.now();

  function commandPending() {
    if (pendingId === null) return false;
    const acked = snap && snap.acked && typeof snap.acked.id === 'number' ? snap.acked.id : -1;
    if (acked >= pendingId) {
      pendingId = null;
      return false;
    }
    return Date.now() - pendingAt < ACK_TIMEOUT_MS;
  }

  const clamp01 = (x) => Math.min(1, Math.max(0, x));
  const fmtInt = (x) => (typeof x === 'number' ? String(x) : '-');
  const fmtNum = (x, digits) => (typeof x === 'number' ? x.toFixed(digits) : '-');
  const clock = (iso) => {
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? '' : d.toLocaleTimeString([], { hour12: false });
  };
  const secondsSince = (iso) => {
    const t = Date.parse(iso || '');
    return Number.isNaN(t) ? null : Math.max(0, (Date.now() - t) / 1000);
  };

  function make(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function chip(text, none) {
    return make('span', none ? 'chip is-none' : 'chip', text);
  }

  function setImage(img, url) {
    if (!url) {
      img.removeAttribute('src');
      img.dataset.src = '';
      img.hidden = true;
      return;
    }
    if (img.dataset.src === url) return;
    img.dataset.src = url;
    img.onload = () => { img.hidden = false; };
    img.onerror = () => { img.hidden = true; };
    img.src = url;
  }

  function scoreBar(score, threshold, blue) {
    const wrap = make('div', 'score');
    const head = make('div', 'score-head');
    head.append(
      make('span', 'label', 'anomaly score'),
      make('span', 'score-num', fmtNum(score, 3)),
      make('span', 'thr-num', threshold == null ? 'threshold -' : `threshold ${threshold.toFixed(3)}`),
    );
    const bar = make('div', 'bar');
    const fill = make('div', 'fill ' + (blue ? 'is-blue' : 'is-green'));
    const marker = make('div', 'marker');
    marker.hidden = true;
    bar.append(fill, marker);
    if (score == null) {
      fill.style.width = '0%';
    } else {
      // The scale runs to twice the threshold, so the threshold sits at the middle mark.
      const top = Math.max((threshold ?? 0) * 2, score * 1.1, 1e-6);
      fill.style.width = (clamp01(score / top) * 100).toFixed(1) + '%';
      if (threshold != null) {
        marker.hidden = false;
        marker.style.left = (clamp01(threshold / top) * 100).toFixed(1) + '%';
      }
    }
    wrap.append(head, bar);
    return wrap;
  }

  function camCard(title, serial, cam) {
    const label = cam.label || 'UNKNOWN';
    const blue = label === 'GOOD';
    const card = make('figure', 'cam ' + (blue ? 'is-blue' : 'is-green') + (label === 'UNCERTAIN' ? ' is-safe' : ''));
    const head = make('div', 'cam-head');
    head.append(make('span', 'cam-title', title), make('span', 'cam-serial', serial || ''));
    const media = make('div', 'media cam-media');
    const img = document.createElement('img');
    img.alt = `${title} brick crop`;
    media.append(img, make('div', 'media-empty', 'no crop'));
    setImage(img, cam.crop_url);
    const verdict = make('div', 'cam-verdict', label);
    const reasons = cam.reasons || [];
    const chips = reasons.length
      ? reasons.map((r) => chip(REASON_LABELS[r] || r, false))
      : [chip('no reason flagged', true)];
    card.append(head, media, verdict, scoreBar(cam.score, cam.threshold, blue), make('div', 'cam-reasons', null));
    card.lastChild.append(...chips);
    return card;
  }

  function renderCameras(insp) {
    const cams = insp.cameras || [];
    if (!cams.length) {
      el.cams.replaceChildren(camCard('Brick crop', '', {
        label: insp.label, score: insp.score, threshold: insp.threshold,
        crop_url: insp.crop_url, reasons: insp.reasons,
      }));
      return;
    }
    el.cams.replaceChildren(...cams.map((cam) => camCard(
      VIEW_TITLES[cam.view] || cam.view || 'Camera', cam.camera, cam)));
  }

  function renderInspection(insp) {
    if (!insp) {
      el.verdict.className = 'verdict is-idle';
      el.verdictPlate.textContent = 'WAITING';
      el.verdictSafe.textContent = '';
      el.verdictSub.textContent = 'no brick inspected yet';
      el.inspMeta.textContent = '';
      el.cams.replaceChildren();
      el.reasons.replaceChildren();
      el.device.textContent = '-';
      el.geometry.textContent = '';
      return;
    }
    const blue = insp.plate === 'BLUE';
    el.verdict.className = 'verdict ' + (blue ? 'is-blue' : 'is-green');
    el.verdictPlate.textContent = `${insp.plate} PLATE`;
    el.verdictSafe.textContent = insp.safe ? ' (safe)' : '';
    const views = (insp.cameras || []).length;
    const fused = views ? `fused from ${views} cameras` : 'single view';
    el.verdictSub.textContent = `${insp.label || 'UNKNOWN'} · ${fused} · cycle ${fmtInt(insp.cycle)} · route ${insp.route || '-'}`;
    el.inspMeta.textContent = insp.t ? `at ${clock(insp.t)}` : '';
    renderCameras(insp);

    const reasons = insp.reasons || [];
    el.reasons.replaceChildren(
      ...(reasons.length
        ? reasons.map((r) => chip(REASON_LABELS[r] || r, false))
        : [chip('no reasons reported', true)]),
    );
    // With per-camera cards, the reasons are shown on each card instead.
    el.reasons.hidden = views > 0;

    const device = DEVICE_NAMES[insp.device] || insp.device || 'device not reported';
    const backend = insp.backend ? `${insp.backend} on ` : '';
    const latency = insp.latency_ms == null ? '' : `, ${Math.round(insp.latency_ms)} ms`;
    el.device.textContent = `${backend}${device}${latency}`;

    const geometry = insp.geometry || {};
    el.geometry.textContent = Object.entries(geometry)
      .filter(([, value]) => typeof value === 'number')
      .slice(0, 4)
      .map(([key, value]) => `${key} ${Number.isInteger(value) ? value : value.toFixed(2)}`)
      .join(' · ');
  }

  // The server reports handover, align and sort only for the current cycle (see RunReducer._current),
  // so this line never shows an earlier cycle's sort. The sort comes first: on a narrow screen the
  // line wraps, and the plate is what the operator reads.
  function renderDetail(s) {
    const parts = [];
    if (s.sort) {
      const plate = s.sort.plate ? `${s.sort.plate} PLATE` : (s.sort.route || 'route -');
      let status = '';
      if (s.sort.active) status = 'running';
      else if (s.sort.ok === true) status = 'done';
      else if (s.sort.ok === false) status = 'failed';
      parts.push(`sort to ${plate} ${status}`.trim());
    }
    if (s.handover) {
      const kept = s.handover.model_kept === true ? ', model kept' : '';
      parts.push(`handover ${s.handover.phase || ''} ${fmtNum(s.handover.duration_s, 1)} s${kept}`);
    }
    if (s.align) {
      parts.push(`align ${fmtInt(s.align.iteration)} dx ${fmtInt(s.align.dx)} dy ${fmtInt(s.align.dy)} px`);
    }
    el.detail.textContent = parts.join(' · ');
  }

  function renderCounts(c) {
    el.cCycles.textContent = fmtInt(c.cycles);
    el.cGood.textContent = fmtInt(c.good);
    el.cDefective.textContent = fmtInt(c.defective);
    el.cUncertain.textContent = fmtInt(c.uncertain);
    el.cMean.textContent = typeof c.mean_cycle_s === 'number' ? `${c.mean_cycle_s.toFixed(1)} s` : '-';
  }

  function renderHistory(history) {
    const sig = history
      .map((h) => [h.cycle, h.label, h.plate, h.safe, h.crop_url].join('|'))
      .join(';');
    if (sig === historySig) return;
    historySig = sig;
    const pad = HISTORY_SLOTS - history.length;
    const tiles = [];
    for (let i = 0; i < HISTORY_SLOTS; i += 1) {
      const tile = make('div', 'tile');
      const h = i >= pad ? history[i - pad] : null;
      if (!h) {
        tile.classList.add('empty');
        tiles.push(tile);
        continue;
      }
      tile.classList.add(h.plate === 'BLUE' ? 'is-blue' : 'is-green');
      if (h.safe) tile.classList.add('is-safe');
      if (h.crop_url) {
        const img = document.createElement('img');
        img.alt = `cycle ${fmtInt(h.cycle)} crop`;
        img.onerror = () => img.remove();
        img.src = h.crop_url;
        tile.append(img);
      }
      tile.append(make('div', 'tag', `C${fmtInt(h.cycle)} ${h.plate}${h.safe ? ' (safe)' : ''}`));
      tiles.push(tile);
    }
    el.strip.replaceChildren(...tiles);
  }

  function renderCommandState(s) {
    const issued = s.control && s.control.issued;
    const acked = s.acked;
    let text = 'no command sent yet';
    let cls = '';
    if (cmdError) {
      text = cmdError;
      cls = 'is-bad';
    } else if (issued) {
      if (acked && typeof acked.id === 'number' && acked.id >= issued.id) {
        text = `loop acked ${issued.cmd} #${issued.id}`;
        cls = 'is-ok';
      } else {
        text = `${issued.cmd} #${issued.id} written, waiting for the loop to ack`;
        cls = 'is-warn';
      }
    }
    el.cmdState.textContent = text;
    el.cmdState.className = 'cmd-state ' + cls;
  }

  function updateButtons() {
    const hasRun = Boolean(snap && snap.has_run);
    const blocked = cmdBusy || commandPending();
    el.btnStart.disabled = !hasRun || blocked;
    el.btnNext.disabled = !hasRun || blocked;
    el.btnStop.disabled = !hasRun;
  }

  function renderAlert(s) {
    const show = Boolean(s.error) && ['HOLD', 'STOPPED', 'RUN_ENDED'].includes(s.stage);
    el.alert.hidden = !show;
    if (show) {
      el.alert.textContent = `Error in ${s.error.where || 'loop'}: ${s.error.message || '(no message)'}`;
    }
  }

  // ----- v5 panels: scan, decision and policy steps, verify, stats and intel (LOOP-CONTRACT sections 8 and 9) -----
  const PATH_NAMES = { direct: 'direct', inspected: 'inspected', fail_safe: 'fail-safe' };
  const BRICK_VERDICTS = { CONFIDENT_DEFECT: 'CONF. DEFECT', GOOD: 'GOOD', DEFECTIVE: 'DEFECTIVE', UNCERTAIN: 'UNCERTAIN' };
  const ms = (x) => (typeof x === 'number' ? `${x.toFixed(1)} ms` : '-');
  const pct = (x) => (typeof x === 'number' ? `${Math.round(x * 100)} %` : '-');
  const numText = (x, digits) => (typeof x === 'number' ? x.toFixed(digits) : '-');

  function brickCard(b, target) {
    const cls = ['brick'];
    if (target) cls.push('is-target');
    if (b.direct_green) cls.push('is-direct');
    const card = make('figure', cls.join(' '));
    const media = make('div', 'media brick-media');
    const img = document.createElement('img');
    img.alt = `brick ${b.id || ''} crop`;
    media.append(img, make('div', 'media-empty', 'no crop'));
    setImage(img, b.crop_url);
    const head = make('div', 'brick-head');
    head.append(
      make('span', 'brick-id', target ? `TARGET ${b.id}` : `brick ${b.id ?? '-'}`),
      make('span', 'brick-verdict', BRICK_VERDICTS[b.verdict] || b.verdict || '-'),
    );
    const reasons = b.reasons || [];
    const reasonRow = make('div', 'brick-reasons');
    reasonRow.append(...(reasons.length
      ? reasons.map((r) => chip(REASON_LABELS[r] || r, false))
      : [chip('no reason flagged', true)]));
    card.append(
      media,
      head,
      make('div', 'brick-facts', `${b.colour || '-'} · ${b.shape || '-'} · tape ${numText(b.tape_score, 2)}`),
      reasonRow,
      make('div', b.direct_green ? 'brick-route is-green' : 'brick-route', b.route || '-'),
    );
    return card;
  }

  function renderScan(scan) {
    if (!scan) {
      el.bricks.replaceChildren(make('div', 'empty-note', 'no scan yet in this run'));
      el.scanMeta.textContent = '';
      return;
    }
    const bricks = scan.bricks || [];
    const how = scan.target_how ? ` · target by ${scan.target_how}` : '';
    el.scanMeta.textContent = `cycle ${fmtInt(scan.cycle)} · ${bricks.length} brick${bricks.length === 1 ? '' : 's'}${how}`;
    if (!bricks.length) {
      el.bricks.replaceChildren(make('div', 'empty-note', 'the scan saw no brick'));
      return;
    }
    el.bricks.replaceChildren(...bricks.map((b) => brickCard(b, scan.target_id != null && b.id === scan.target_id)));
  }

  function renderPolicy(p) {
    const cur = p && p.current;
    const dec = cur && cur.decision;
    if (!dec) {
      el.decisionPlate.textContent = 'WAITING';
      el.decisionPlate.className = 'decision-plate';
      el.decisionRule.textContent = 'no decision in this cycle yet';
    } else {
      el.decisionPlate.textContent = `${dec.plate || '-'} PLATE`;
      el.decisionPlate.className = 'decision-plate ' + (dec.plate === 'BLUE' ? 'is-blue' : 'is-green');
      el.decisionRule.textContent = `rule ${dec.rule || '-'} · path ${PATH_NAMES[dec.via] || dec.path || '-'} · brick ${dec.target_id ?? '-'}`;
    }
    const tri = cur && cur.triage;
    el.policyMeta.textContent = tri ? `triage: ${tri.route_text || tri.route || '-'}` : '';

    const phases = (cur && cur.phases) || [];
    if (!phases.length) {
      el.timeline.replaceChildren(make('li', 'step empty', 'no policy steps in this cycle'));
    } else {
      el.timeline.replaceChildren(...phases.map((ph, i) => {
        const li = make('li', 'step' + (ph.phase === 'done' ? ' is-done' : ''));
        const device = DEVICE_NAMES[ph.device] || ph.device || 'device not reported';
        li.append(
          make('span', 'step-name', ph.phase),
          make('span', 'step-dev', device),
          make('span', 'step-model', ph.model_id || '-'),
          make('span', 'step-ms', ph.step_ms == null ? (i === 0 ? 'start' : '-') : ms(ph.step_ms)),
        );
        return li;
      }));
    }

    const v = cur && cur.verify;
    if (!v) {
      el.verify.className = 'verify';
      el.verify.textContent = 'verify: not reached yet';
      el.verifyFrame.hidden = true;
    } else {
      el.verify.className = 'verify ' + (v.ok ? 'is-ok' : 'is-bad');
      el.verify.textContent = `verify ${v.ok ? 'OK' : 'FAILED'}: intended ${v.plate_intended || '-'}, released ${v.release_side || '-'}`;
      el.verifyFrame.hidden = !v.frame_top_url;
      setImage(el.verifyImg, v.frame_top_url || null);
    }
  }

  function metric(label, value, source) {
    const wrap = make('div', 'metric');
    wrap.append(make('dt', null, label), make('dd', null, value), make('small', null, source));
    return wrap;
  }

  function benchRow(r) {
    const row = make('tr', r.error ? 'is-err' : '');
    const energy = r.energy_per_inference_j == null ? 'not measured' : `${numText(r.energy_per_inference_j, 4)} J`;
    row.append(
      make('td', null, r.device),
      make('td', null, r.model),
      make('td', 'num', r.error ? r.error : numText(r.p50_ms, 2)),
      make('td', 'num', numText(r.p95_ms, 2)),
      make('td', 'num', numText(r.throughput_ips, 1)),
      make('td', 'num', numText(r.temp_c_mean, 1)),
      make('td', null, energy),
    );
    return row;
  }

  function latencyTile(label, stats, source) {
    const value = stats && stats.p50 != null ? `${Math.round(stats.p50)} / ${Math.round(stats.p95)} ms` : '-';
    return metric(label, value, `${source} · n ${stats ? stats.n : 0}`);
  }

  function renderIntel(intel, p) {
    const t = intel && intel.telemetry;
    if (!t) {
      el.intelMeta.textContent = 'no telemetry folder';
      el.intelLive.replaceChildren();
    } else if (!t.available) {
      el.intelMeta.textContent = t.reason || 'no telemetry yet';
      el.intelLive.replaceChildren();
    } else {
      el.intelMeta.textContent = `telemetry ${numText(t.age_s, 1)} s old${t.stale ? ' (stale)' : ''}`;
      el.intelLive.replaceChildren(
        metric('temp pkg', `${numText(t.temp_c.pkg, 1)} °C`, `temp_c.pkg · 60 s avg ${numText(t.temp_c.avg_60s_pkg, 1)}`),
        metric('cpu util', pct(t.cpu_util.latest), `cpu_util (/proc/stat) · 60 s avg ${pct(t.cpu_util.avg_60s)}`),
        metric('gpu freq', t.gpu.freq_mhz == null ? '-' : `${Math.round(t.gpu.freq_mhz)} MHz`,
          `gpu freq ratio ${pct(t.gpu.freq_ratio)} (proxy: busy not readable)`),
        metric('npu busy', pct(t.npu.busy), `npu accel0_busy_us · 60 s avg ${pct(t.npu.avg_60s_busy)}`),
        metric('power', t.power.available ? `${numText(t.power.pkg_w, 1)} W` : 'not measured',
          'pkg_w null: RAPL is root-only here'),
        metric('memory', pct(t.mem.used_pct == null ? null : t.mem.used_pct / 100), 'mem.used_pct'),
      );
    }

    // Latency tiles come from events.jsonl only (no telemetry needed); the text line below them is not used.
    const lat = p && p.latency;
    el.intelLat.hidden = true;
    if (lat) {
      el.intelLive.append(
        latencyTile('step p50 / p95', lat.step_ms, 'policy_run times'),
        latencyTile('cycle p50 / p95', lat.cycle_ms, 'decision to verify'),
        latencyTile('inspect p50 / p95', lat.inspection_ms, 'inspection latency_ms'),
      );
    }

    const bench = intel && intel.bench;
    if (!bench || !bench.available) {
      el.bench.replaceChildren();
      el.intelSrc.textContent = bench ? `${bench.reason}: ${bench.file}` : 'no benchmark file configured';
      return;
    }
    const head = make('tr');
    ['device', 'model', 'p50 ms', 'p95 ms', 'img/s', 'temp C', 'energy'].forEach((h) => head.append(make('th', null, h)));
    el.bench.replaceChildren(make('thead'), make('tbody'));
    el.bench.querySelector('thead').append(head);
    el.bench.querySelector('tbody').append(...bench.rows.map(benchRow));
    el.intelSrc.textContent = `source: bench.json ${bench.file} · status ${bench.status || '-'} · energy ${bench.energy_note} · power ${bench.power_note || '-'}`;
  }

  function renderStats(s) {
    const p = s.policy || {};
    const v = p.verify || {};
    el.sSuccess.textContent = v.total ? pct(v.success_rate) : '-';
    el.sSuccess.title = v.total ? `${v.ok} of ${v.total} cycles verified` : 'no verify yet';
    el.sErrors.textContent = fmtInt(s.errors);
    const routes = p.routes || {};
    el.sRoute.textContent = `${fmtInt(routes.GREEN_DIRECT)} / ${fmtInt(routes.INSPECT)}`;
  }

  // The pre-grasp frame is a third card in the verdict grid (renderInspection rebuilds that grid each snapshot).
  function renderFrame(insp) {
    const url = insp && insp.frame_url;
    el.frameCard.hidden = !url;
    setImage(el.frameImg, url || null);
    el.cams.classList.toggle('with-frame', Boolean(url));
    if (url) el.cams.append(el.frameCard);
  }

  function render(s) {
    snap = s;
    const stage = s.stage || 'NO_RUN';
    document.body.dataset.stage = stage;
    const [label, sub] = STAGES[stage] || [stage, ''];
    el.stage.textContent = label;
    el.stageSub.textContent = (stage === 'HOLD' || stage === 'STOPPED') && s.error
      ? `${s.error.where || 'error'}: ${s.error.message || ''}`
      : sub;
    el.runId.textContent = s.has_run ? `run ${s.run_id}` : 'no run';
    el.liveSource.textContent = s.live_source || 'wrist camera (live.jpg)';
    renderMode(s.mode);
    renderAlert(s);
    renderDetail(s);
    renderInspection(s.inspection);
    renderFrame(s.inspection);
    renderCounts(s.counts || {});
    renderStats(s);
    renderScan(s.policy && s.policy.scan);
    renderPolicy(s.policy);
    renderIntel(s.intel, s.policy);
    renderHistory(s.history || []);
    renderCommandState(s);
    updateButtons();
  }

  function renderMode(mode) {
    const badge = el.modeBadge;
    badge.className = 'badge';
    if (mode === 'dry-run') {
      badge.textContent = 'dry run, no robot motion';
      badge.classList.add('is-dry');
    } else if (mode === 'live') {
      badge.textContent = 'live, robot armed';
      badge.classList.add('is-live');
    } else {
      badge.textContent = mode ? `mode ${mode}` : 'mode not reported';
    }
  }

  function startLivePreview() {
    const img = el.live;
    const next = () => {
      img.src = '/run/current/live.jpg?ts=' + Date.now();
    };
    img.onload = () => {
      img.hidden = false;
      setTimeout(next, LIVE_GAP_MS);
    };
    img.onerror = () => {
      img.hidden = true;
      setTimeout(next, 500);
    };
    next();
  }

  function renderLink() {
    let text = 'connecting';
    let cls = '';
    // A dead server leaves EventSource in CONNECTING (it retries), so the state alone never says offline.
    const quietMs = Date.now() - (lastSnapshotMs || pageStartMs);
    if (es.readyState !== EventSource.OPEN && quietMs > SERVER_DOWN_MS) {
      text = 'server offline';
      cls = 'is-bad';
    } else if (!snap) {
      text = 'connecting';
    } else if (es.readyState !== EventSource.OPEN) {
      text = 'reconnecting';
      cls = 'is-warn';
    } else if (!snap.has_run) {
      text = 'waiting for a run';
      cls = 'is-warn';
    } else {
      const age = secondsSince(snap.heartbeat_t);
      if (age === null) {
        text = 'no heartbeat yet';
        cls = 'is-warn';
      } else if (age <= LIVE_MAX_AGE_S) {
        text = 'loop live';
        cls = 'is-ok';
      } else {
        text = `loop silent ${Math.round(age)} s`;
        cls = age > 10 ? 'is-bad' : 'is-warn';
      }
    }
    el.linkBadge.textContent = text;
    el.linkBadge.className = 'badge ' + cls;
  }

  function tick() {
    renderLink();
    const sec = snap && snap.has_run ? secondsSince(snap.state_t) : null;
    el.stageTime.textContent = sec === null ? '' : `${sec.toFixed(1)} s in stage`;
    updateButtons();
  }

  async function sendCommand(cmd) {
    cmdBusy = true;
    updateButtons();
    try {
      const res = await fetch('/api/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cmd }),
        cache: 'no-store',
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok || body.ok !== true) {
        throw new Error(body.error || `HTTP ${res.status}`);
      }
      cmdError = '';
      pendingId = body.id;
      pendingAt = Date.now();
      el.cmdState.textContent = `${cmd} #${body.id} written, waiting for the loop to ack`;
      el.cmdState.className = 'cmd-state is-warn';
    } catch (err) {
      cmdError = `${cmd} not sent: ${err.message}`;
      el.cmdState.textContent = cmdError;
      el.cmdState.className = 'cmd-state is-bad';
    } finally {
      cmdBusy = false;
      updateButtons();
    }
  }

  el.btnStart.addEventListener('click', () => sendCommand('start'));
  el.btnNext.addEventListener('click', () => sendCommand('next'));
  el.btnStop.addEventListener('click', () => sendCommand('stop'));

  // Video mode: larger type and no debug lines. The state is in the URL (?video=1) so a screen recording can reload it.
  function setVideo(on) {
    document.body.classList.toggle('video', on);
    el.btnVideo.setAttribute('aria-pressed', on ? 'true' : 'false');
    el.btnVideo.textContent = on ? 'video mode on' : 'video mode';
  }
  el.btnVideo.addEventListener('click', () => {
    const on = !document.body.classList.contains('video');
    setVideo(on);
    const url = new URL(window.location.href);
    if (on) url.searchParams.set('video', '1');
    else url.searchParams.delete('video');
    window.history.replaceState(null, '', url.pathname + url.search);
  });
  setVideo(new URLSearchParams(window.location.search).get('video') === '1');

  const es = new EventSource('/api/events');
  es.addEventListener('snapshot', (event) => {
    try {
      render(JSON.parse(event.data));
      lastSnapshotMs = Date.now();
    } catch (err) {
      console.error('dashboard: bad snapshot', err);
    }
  });

  startLivePreview();
  renderHistory([]);
  setInterval(tick, TICK_MS);
  tick();
})();
