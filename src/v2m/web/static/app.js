// v2m web UI. Plain JS, no build step (docs/ARCHITECTURE.md M7).
"use strict";

const $ = (id) => document.getElementById(id);
const PHASES = [
  ["ingest", "Picking sharp frames"],
  ["sfm_sparse", "Solving camera positions"],
  ["sfm_dense", "Estimating depth"],
  ["mesh", "Building the surface"],
  ["print_prep", "Making it printable"],
];
const ICONS = { complete: "✓", failed: "✕", skipped: "↷", running: "", pending: "" };

let selectedFile = null;
let currentRun = null;
let source = null;
let runningSince = {};
let timer = null;

// -- new run form -------------------------------------------------------------------

function chooseFile(file) {
  selectedFile = file;
  $("drop-label").textContent = file ? `${file.name} (${(file.size / 1e6).toFixed(1)} MB)` : "Drop a video here, or click to choose";
  $("drop").classList.toggle("has-file", !!file);
  $("start").disabled = !file;
}

$("file").addEventListener("change", (e) => chooseFile(e.target.files[0] || null));
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => {
  e.preventDefault(); drop.classList.add("over");
}));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => {
  e.preventDefault(); drop.classList.remove("over");
}));
drop.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) chooseFile(e.dataTransfer.files[0]); });

function showFormError(message) {
  $("form-error").textContent = message || "";
  $("form-error").hidden = !message;
}

$("start").addEventListener("click", () => {
  if (!selectedFile) return;
  showFormError(null);
  const form = new FormData();
  form.append("file", selectedFile);
  form.append("preset", document.querySelector("input[name=preset]:checked").value);
  if ($("aruco").value) form.append("aruco_marker_mm", $("aruco").value);
  if ($("target").value) form.append("target_size_mm", $("target").value);

  // XHR rather than fetch: upload progress matters for multi-GB phone videos.
  const xhr = new XMLHttpRequest();
  const bar = $("upload-progress");
  bar.hidden = false;
  $("start").disabled = true;
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) bar.firstElementChild.style.width = `${(100 * e.loaded) / e.total}%`;
  };
  xhr.onload = () => {
    bar.hidden = true;
    bar.firstElementChild.style.width = "0";
    $("start").disabled = false;
    let body = {};
    try { body = JSON.parse(xhr.responseText); } catch (_) { /* non-JSON error page */ }
    if (xhr.status !== 200) {
      showFormError([body.error || body.detail || `Upload failed (${xhr.status})`, body.remedy].filter(Boolean).join(" — "));
      return;
    }
    chooseFile(null);
    $("file").value = "";
    openRun(body.run_id);
    loadHistory();
  };
  xhr.onerror = () => {
    bar.hidden = true; $("start").disabled = false;
    showFormError("Couldn't reach the v2m server. Is `v2m serve` still running?");
  };
  xhr.open("POST", "/api/jobs");
  xhr.send(form);
});

// -- a run: phases, live events, result ---------------------------------------------

function renderPhases(phases) {
  const list = $("phases");
  list.innerHTML = "";
  for (const [name, friendly] of PHASES) {
    const info = phases.find((p) => p.name === name) || { status: "pending" };
    const li = document.createElement("li");
    li.className = info.status;
    li.dataset.phase = name;
    li.innerHTML = `<span class="icon"></span><span class="label"></span><span class="time"></span>`;
    li.querySelector(".icon").textContent = ICONS[info.status] || "";
    li.querySelector(".label").textContent = friendly;
    li.querySelector(".time").textContent = info.duration_s != null ? formatSeconds(info.duration_s) : "";
    list.appendChild(li);
  }
}

function setPhase(name, status, durationS) {
  const li = document.querySelector(`#phases li[data-phase="${name}"]`);
  if (!li) return;
  li.className = status;
  li.querySelector(".icon").textContent = ICONS[status] || "";
  if (status === "running") runningSince[name] = Date.now();
  if (durationS != null) li.querySelector(".time").textContent = formatSeconds(durationS);
  if (status === "skipped") li.querySelector(".time").textContent = "done earlier";
}

function tickRunning() {
  for (const [name, since] of Object.entries(runningSince)) {
    const li = document.querySelector(`#phases li[data-phase="${name}"]`);
    if (li && li.classList.contains("running")) li.querySelector(".time").textContent = formatSeconds((Date.now() - since) / 1000);
  }
}

function formatSeconds(s) {
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)}s`;
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

function setBadge(status, text) {
  const badge = $("job-badge");
  badge.className = `badge ${status}`;
  badge.textContent = text || status;
}

function appendLog(line) {
  const log = $("log");
  const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 4;
  log.textContent += line + "\n";
  if (atBottom) log.scrollTop = log.scrollHeight;
}

function showFailure(error, remedy) {
  $("failure").hidden = !error;
  $("failure-message").textContent = error || "";
  $("failure-remedy").textContent = remedy ? `→ ${remedy}` : "";
}

async function openRun(runId) {
  if (source) { source.close(); source = null; }
  currentRun = runId;
  runningSince = {};
  $("log").textContent = "";
  $("job").hidden = false;
  $("result").hidden = true;
  $("job-title").textContent = runId;
  const detail = await fetchJSON(`/api/runs/${encodeURIComponent(runId)}`);
  if (!detail || currentRun !== runId) return;
  renderRun(detail);
  const live = detail.job && (detail.job.status === "queued" || detail.job.status === "running");
  if (live) listen(runId);
  $("job").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderRun(detail) {
  renderPhases(detail.phases);
  const jobStatus = detail.job && !["complete", "failed"].includes(detail.job.status) ? detail.job.status : null;
  setBadge(jobStatus || detail.status, jobStatus ? jobStatus : detail.status_text);
  const failed = detail.phases.find((p) => p.status === "failed");
  const jobError = detail.job && detail.job.status === "failed" ? detail.job : null;
  showFailure(failed ? failed.error : jobError && jobError.error, failed ? failed.remedy : jobError && jobError.remedy);
  const ahead = detail.job && detail.job.jobs_ahead;
  $("job-queue").hidden = !(detail.job && detail.job.status === "queued" && ahead);
  $("job-queue").textContent = ahead ? `Waiting for ${ahead} run${ahead > 1 ? "s" : ""} ahead of this one.` : "";
  if (detail.print_report) renderResult(detail);
}

function renderResult(detail) {
  const report = detail.print_report;
  $("result").hidden = false;
  const viewer = $("viewer");
  viewer.setAttribute("src", `${detail.files["model.glb"]}?v=${Date.now()}`);
  const size = report.bbox_mm.map((v) => v.toFixed(1)).join(" × ");
  const rows = [
    ["Size", `${size} mm`],
    ["Volume", `${(report.volume_mm3 / 1000).toFixed(1)} cm³`],
    ["Watertight", report.watertight ? "yes" : "no"],
    ["Scale", report.scale_method === "fit_to_build_volume" ? "fitted to printer (not real-world)" : report.scale_method],
    ["Repair", `rung ${report.repair_rung_used} of 6`],
  ];
  const stats = $("stats");
  stats.innerHTML = "";
  for (const [k, v] of rows) {
    const dt = document.createElement("dt"); dt.textContent = k;
    const dd = document.createElement("dd"); dd.textContent = v;
    stats.append(dt, dd);
  }
  $("dl-stl").href = detail.files["model.stl"];
  $("dl-obj").href = detail.files["model.obj"];
  $("dl-glb").href = detail.files["model.glb"];
  $("report-link").hidden = !detail.report_url;
  if (detail.report_url) $("report-link").href = detail.report_url;
  const warnings = $("warnings");
  warnings.innerHTML = "";
  for (const w of report.warnings || []) {
    const li = document.createElement("li"); li.textContent = w; warnings.appendChild(li);
  }
}

function listen(runId) {
  source = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`);
  source.addEventListener("status", (e) => {
    const data = JSON.parse(e.data);
    setBadge(data.status);
    if (data.status === "complete" || data.status === "failed") {
      source.close(); source = null;
      openRun(runId).then(loadHistory);
    } else if (data.status === "running") {
      $("job-queue").hidden = true;
    }
  });
  source.addEventListener("phase", (e) => {
    const data = JSON.parse(e.data);
    setPhase(data.phase, data.status, data.duration_s);
  });
  source.addEventListener("log", (e) => {
    const data = JSON.parse(e.data);
    appendLog(`${data.level === "INFO" ? "" : data.level + ": "}${data.message}`);
  });
  // EventSource reconnects on its own (sending Last-Event-ID) after a
  // network blip; the server replays only what was missed.
}

$("resume").addEventListener("click", async () => {
  if (!currentRun) return;
  const response = await fetch(`/api/runs/${encodeURIComponent(currentRun)}/resume`, { method: "POST" });
  if (response.ok) { showFailure(null); openRun(currentRun); loadHistory(); }
});

// -- history --------------------------------------------------------------------------

async function loadHistory() {
  const runs = await fetchJSON("/api/runs");
  if (!runs) return;
  const list = $("runs");
  list.innerHTML = "";
  $("history-empty").hidden = runs.length > 0;
  for (const run of runs) {
    const li = document.createElement("li");
    const created = new Date(run.created_at);
    li.innerHTML = `<span class="badge"></span><span class="name"></span><span class="meta"></span>`;
    li.querySelector(".badge").className = `badge ${run.status}`;
    li.querySelector(".badge").textContent = run.status === "complete" ? "done" : run.status;
    li.querySelector(".name").textContent = run.source_name || run.run_id;
    li.querySelector(".meta").textContent = `${run.preset} · ${created.toLocaleString()}`;
    li.addEventListener("click", () => openRun(run.run_id));
    list.appendChild(li);
  }
}

async function fetchJSON(url) {
  try {
    const response = await fetch(url);
    return response.ok ? await response.json() : null;
  } catch (_) {
    return null;
  }
}

timer = setInterval(tickRunning, 1000);
loadHistory();
