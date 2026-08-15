/* Video Understanding — web client.
 *
 * Plain ES modules-free JavaScript against the same REST API the CLI uses. It
 * is deliberately dependency-free: this file is served by the API process
 * itself, so a machine with no network still gets the whole UI.
 *
 * Three long-lived pieces of state, and everything else derives from them:
 *
 *   library  — the paginated list of every upload, mirrored from SQLite
 *   current  — the video being viewed, including its full analysis
 *   player   — the <video> element, which owns playback time
 *
 * The URL hash is the source of truth for which video is open (`#/v/<id>`),
 * so a video is linkable, reloadable and back-buttonable.
 */
"use strict";

/* --- Small helpers ------------------------------------------------------- */

const $ = (id) => document.getElementById(id);
const el = (tag, className, html) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (html != null) node.innerHTML = html;
  return node;
};

const esc = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

/** `73.4` -> `1:13`. Hours appear only when the video has them. */
function clock(seconds) {
  if (!isFinite(seconds) || seconds < 0) seconds = 0;
  const total = Math.floor(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const mm = h ? String(m).padStart(2, "0") : String(m);
  return `${h ? h + ":" : ""}${mm}:${String(s).padStart(2, "0")}`;
}

/** Same, with a tenth of a second — used where frame accuracy is the point. */
const clockPrecise = (seconds) =>
  `${clock(seconds)}.${Math.floor((Math.max(0, seconds) % 1) * 10)}`;

function bytes(value) {
  if (!value) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let n = value;
  let unit = 0;
  while (n >= 1024 && unit < units.length - 1) { n /= 1024; unit += 1; }
  return `${n < 10 && unit > 0 ? n.toFixed(1) : Math.round(n)} ${units[unit]}`;
}

function ago(iso) {
  const then = new Date(iso).getTime();
  if (!isFinite(then)) return "";
  const seconds = Math.max(0, (Date.now() - then) / 1000);
  if (seconds < 60) return "just now";
  const steps = [[60, "min"], [24, "hr"], [7, "day"], [4.35, "wk"], [12, "mo"]];
  let value = seconds / 60;
  let label = "min";
  for (let i = 1; i < steps.length; i += 1) {
    if (value < steps[i][0]) break;
    value /= steps[i][0];
    label = steps[i][1];
  }
  const rounded = Math.floor(value);
  return `${rounded} ${label}${rounded === 1 ? "" : "s"} ago`;
}

const debounce = (fn, wait) => {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), wait);
  };
};

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

async function api(path, options) {
  const response = await fetch(path, options);
  const text = await response.text();
  let data = null;
  if (text) { try { data = JSON.parse(text); } catch { data = null; } }
  if (!response.ok) {
    const detail = (data && (data.detail || data.message)) || `HTTP ${response.status}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

/* --- Toasts -------------------------------------------------------------- */

function toast(message, kind = "info", ms = 4200) {
  const node = el("div", `toast ${kind}`, esc(message));
  $("toasts").appendChild(node);
  setTimeout(() => {
    node.classList.add("out");
    setTimeout(() => node.remove(), 200);
  }, ms);
}

/* --- Application state --------------------------------------------------- */

const state = {
  library: { items: [], total: 0, counts: {}, limit: 24, offset: 0, q: "", status: "", sort: "newest" },
  current: null,          // { video_id, status, result, ... }
  captionsOn: false,
  followPlayback: true,
  zoom: 1,
  statusPoll: null,
  libraryPoll: null,
  activeSceneIndex: -1,
  transcriptRows: [],
  deleteArmed: false,
};

const player = $("player");

/* --- Theme --------------------------------------------------------------- */

const THEME_KEY = "vu.theme";

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem(THEME_KEY, theme); } catch { /* private mode */ }
}

(function initTheme() {
  let stored = null;
  try { stored = localStorage.getItem(THEME_KEY); } catch { /* ignore */ }
  const prefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
  applyTheme(stored || (prefersLight ? "light" : "dark"));
})();

$("btn-theme").addEventListener("click", () =>
  applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));

/* --- Health and limits --------------------------------------------------- */

async function loadHealth() {
  const node = $("health");
  const label = node.querySelector(".health-text");
  try {
    const health = await api("/health");
    const unavailable = Object.entries(health.providers || {})
      .filter(([, value]) => !String(value).startsWith("ok"))
      .map(([key]) => key);

    node.className = `health is-${health.status === "ok" ? "ok" : "degraded"}`;
    label.textContent = unavailable.length
      ? `${unavailable.join(", ")} unavailable`
      : `all systems ok · v${health.version}`;
    node.title = Object.entries(health.providers || {})
      .map(([key, value]) => `${key}: ${value}`).join("\n");
  } catch {
    node.className = "health is-down";
    label.textContent = "API unreachable";
  }
}

/* --- Library ------------------------------------------------------------- */

const STATUS_FILTERS = [
  { key: "", label: "All" },
  { key: "completed", label: "Ready" },
  { key: "processing", label: "Running" },
  { key: "queued", label: "Queued" },
  { key: "failed", label: "Failed" },
];

function renderFilters() {
  const counts = state.library.counts || {};
  const container = $("filters");
  container.innerHTML = "";
  for (const filter of STATUS_FILTERS) {
    const count = filter.key ? counts[filter.key] : counts.all;
    if (filter.key && !count && state.library.status !== filter.key) continue;
    const chip = el("button", `chip${state.library.status === filter.key ? " active" : ""}`,
      `${esc(filter.label)}<b>${count ?? 0}</b>`);
    chip.addEventListener("click", () => {
      state.library.status = filter.key;
      state.library.offset = 0;
      loadLibrary();
    });
    container.appendChild(chip);
  }
}

function libraryItemNode(item) {
  const node = el("button", `lib-item${state.current && state.current.video_id === item.video_id ? " active" : ""}`);
  node.setAttribute("role", "listitem");
  node.dataset.id = item.video_id;

  const thumb = el("div", "lib-thumb");
  if (item.status === "completed") {
    const img = el("img");
    img.loading = "lazy";
    img.alt = "";
    img.src = `/v1/videos/${item.video_id}/thumbnail`;
    img.addEventListener("error", () => img.remove());
    thumb.appendChild(img);
  } else {
    thumb.appendChild(el("span", "ph",
      '<svg viewBox="0 0 24 24"><rect x="2" y="4" width="20" height="16" rx="3"/></svg>'));
  }
  if (item.duration) thumb.appendChild(el("span", "dur", clock(item.duration)));

  const meta = el("div", "lib-meta");
  meta.appendChild(el("div", "lib-name", esc(item.title || item.filename)));

  const sub = el("div", "lib-sub");
  sub.appendChild(el("span", `status-pill status-${item.status}`, esc(statusLabel(item))));
  sub.appendChild(el("span", "sep", "·"));
  sub.appendChild(el("span", null, esc(ago(item.created_at))));
  if (item.status === "completed" && item.scene_count) {
    sub.appendChild(el("span", "sep", "·"));
    sub.appendChild(el("span", null, `${item.scene_count} scenes`));
  }
  meta.appendChild(sub);

  node.appendChild(thumb);
  node.appendChild(meta);

  if (item.status === "processing" || item.status === "queued") {
    const bar = el("div", "lib-progress");
    bar.appendChild(el("i")).style.width = `${Math.round((item.progress || 0) * 100)}%`;
    node.appendChild(bar);
  }

  node.addEventListener("click", () => openVideo(item.video_id));
  return node;
}

function statusLabel(item) {
  if (item.status === "processing") return (item.stage || "processing").replace(/_/g, " ");
  if (item.status === "completed") return "ready";
  return item.status;
}

function renderLibrary() {
  const container = $("library");
  container.innerHTML = "";

  if (!state.library.items.length) {
    const message = state.library.q || state.library.status
      ? "No videos match those filters."
      : "Nothing here yet. Upload a video to begin.";
    container.appendChild(el("div", "lib-empty", esc(message)));
  } else {
    for (const item of state.library.items) container.appendChild(libraryItemNode(item));
  }

  const shown = state.library.items.length;
  $("library-count").textContent = state.library.total
    ? `${shown} of ${state.library.total} video${state.library.total === 1 ? "" : "s"}`
    : "";
  $("btn-more").classList.toggle("hidden", shown >= state.library.total);
  renderFilters();
  scheduleLibraryPoll();
}

function librarySkeleton() {
  const container = $("library");
  container.innerHTML = "";
  for (let i = 0; i < 5; i += 1) {
    const row = el("div", "lib-item");
    row.appendChild(el("div", "lib-thumb skeleton"));
    const meta = el("div", "lib-meta");
    meta.appendChild(el("div", "skeleton")).style.cssText = "height:12px;width:80%";
    meta.appendChild(el("div", "skeleton")).style.cssText = "height:10px;width:55%;margin-top:6px";
    row.appendChild(meta);
    container.appendChild(row);
  }
}

async function loadLibrary({ append = false, silent = false } = {}) {
  const { limit, offset, q, status, sort } = state.library;
  if (!append && !silent && !state.library.items.length) librarySkeleton();

  const params = new URLSearchParams({ limit, offset: append ? offset : 0, sort });
  if (q) params.set("q", q);
  if (status) params.set("status", status);

  try {
    const page = await api(`/v1/library?${params}`);
    state.library.items = append ? state.library.items.concat(page.items) : page.items;
    state.library.total = page.total;
    state.library.counts = page.counts;
    state.library.offset = append ? offset : 0;
    renderLibrary();
  } catch (error) {
    if (!silent) $("library").innerHTML =
      `<div class="lib-empty err">Could not load the library: ${esc(error.message)}</div>`;
  }
}

/** Keep polling only while something is actually in flight. */
function scheduleLibraryPoll() {
  const busy = state.library.items.some((item) =>
    item.status === "queued" || item.status === "processing");
  clearTimeout(state.libraryPoll);
  if (busy) state.libraryPoll = setTimeout(() => loadLibrary({ silent: true }), 1500);
}

$("search").addEventListener("input", debounce((event) => {
  state.library.q = event.target.value.trim();
  state.library.offset = 0;
  loadLibrary();
}, 220));

$("sort").addEventListener("change", (event) => {
  state.library.sort = event.target.value;
  state.library.offset = 0;
  loadLibrary();
});

$("btn-refresh").addEventListener("click", () => loadLibrary());
$("btn-more").addEventListener("click", () => {
  state.library.offset += state.library.limit;
  loadLibrary({ append: true });
});

/* --- Upload -------------------------------------------------------------- */

const fileInput = $("file");
$("btn-upload-top").addEventListener("click", () => fileInput.click());
$("btn-upload-empty").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (event) => {
  uploadAll(Array.from(event.target.files || []));
  fileInput.value = "";
});

let dragDepth = 0;
window.addEventListener("dragenter", (event) => {
  if (!Array.from(event.dataTransfer?.types || []).includes("Files")) return;
  dragDepth += 1;
  $("dropzone").classList.add("on");
});
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) $("dropzone").classList.remove("on");
});
window.addEventListener("drop", (event) => {
  event.preventDefault();
  dragDepth = 0;
  $("dropzone").classList.remove("on");
  uploadAll(Array.from(event.dataTransfer?.files || []));
});

async function uploadAll(files) {
  if (!files.length) return;
  let first = null;
  for (const file of files) {
    const id = await uploadOne(file);
    if (id && !first) first = id;
  }
  await loadLibrary();
  if (first) openVideo(first);
}

async function uploadOne(file) {
  const body = new FormData();
  body.append("file", file);
  try {
    const data = await api("/v1/videos", { method: "POST", body });
    toast(`${file.name} queued for analysis`, "ok");
    return data.video_id;
  } catch (error) {
    toast(`${file.name}: ${error.message}`, "err", 7000);
    return null;
  }
}

/* --- Views and routing --------------------------------------------------- */

function showView(name) {
  for (const view of ["empty", "progress", "detail"]) {
    $(`view-${view}`).classList.toggle("hidden", view !== name);
  }
}

function currentRoute() {
  const match = /^#\/v\/([A-Za-z0-9_-]+)/.exec(location.hash);
  return match ? match[1] : null;
}

function openVideo(videoId) {
  if (currentRoute() === videoId) { route(); return; }
  location.hash = `#/v/${videoId}`;
}

window.addEventListener("hashchange", route);

async function route() {
  const videoId = currentRoute();
  clearTimeout(state.statusPoll);

  if (!videoId) {
    state.current = null;
    stopPlayback();
    showView("empty");
    renderLibrary();
    return;
  }

  try {
    const video = await api(`/v1/videos/${videoId}`);
    state.current = video;
    markActiveInLibrary(videoId);

    if (video.status === "completed" && video.result) {
      showView("detail");
      renderDetail(video);
    } else if (video.status === "failed") {
      showView("progress");
      renderProgress(video);
    } else {
      showView("progress");
      renderProgress(video);
      pollStatus(videoId);
    }
  } catch (error) {
    toast(error.message, "err");
    location.hash = "";
  }
}

function markActiveInLibrary(videoId) {
  for (const node of document.querySelectorAll(".lib-item")) {
    node.classList.toggle("active", node.dataset.id === videoId);
  }
}

/* --- Progress view ------------------------------------------------------- */

const STAGES = [
  ["validating", "Validating the file"],
  ["extracting_metadata", "Reading metadata"],
  ["extracting_audio", "Extracting audio"],
  ["transcribing", "Transcribing speech"],
  ["detecting_scenes", "Detecting scenes"],
  ["extracting_frames", "Sampling frames"],
  ["analyzing_frames", "Describing frames"],
  ["reasoning", "Reasoning over the video"],
  ["persisting", "Saving results"],
];

function renderProgress(video) {
  const failed = video.status === "failed";
  $("progress-title").textContent = failed ? "Processing failed" : "Analysing video";
  $("progress-file").textContent = video.filename || "";

  const percent = Math.round((video.progress || 0) * 100);
  const ring = $("progress-ring");
  // A failed job keeps the progress it actually reached; filling the ring
  // would read as "finished", which is the opposite of what happened.
  ring.style.setProperty("--p", percent);
  ring.classList.toggle("is-failed", failed);
  $("progress-pct").textContent = failed ? "!" : `${percent}%`;

  const reached = STAGES.findIndex(([key]) => key === video.stage);
  const list = $("stages");
  list.innerHTML = "";
  STAGES.forEach(([, label], index) => {
    const done = reached > index || video.status === "completed";
    const here = reached === index;
    const item = el("li", [
      done ? "done" : "",
      here && !failed ? "current" : "",
      here && failed ? "failed" : "",
    ].filter(Boolean).join(" "));
    item.appendChild(el("span", "bullet"));
    item.appendChild(el("span", null, esc(label)));
    list.appendChild(item);
  });

  $("progress-hint").classList.toggle("hidden", failed);
  $("progress-actions").classList.toggle("hidden", !failed);

  const error = $("progress-error");
  error.classList.toggle("hidden", !failed);
  if (failed) error.textContent = video.error || "The pipeline reported an unknown failure.";
}

$("btn-retry").addEventListener("click", () => fileInput.click());
$("btn-discard").addEventListener("click", async () => {
  const videoId = currentRoute();
  if (!videoId) return;
  try {
    await api(`/v1/videos/${videoId}`, { method: "DELETE" });
    toast("Removed from the library", "ok");
    location.hash = "";
    await loadLibrary();
  } catch (error) {
    toast(`Could not remove it: ${error.message}`, "err");
  }
});

function pollStatus(videoId) {
  clearTimeout(state.statusPoll);
  state.statusPoll = setTimeout(async () => {
    if (currentRoute() !== videoId) return;
    try {
      const status = await api(`/v1/videos/${videoId}/status`);
      if (status.status === "completed") {
        loadLibrary({ silent: true });
        route();
        return;
      }
      renderProgress({ ...status, filename: state.current?.filename });
      if (status.status === "failed") {
        loadLibrary({ silent: true });
        return;
      }
      pollStatus(videoId);
    } catch {
      pollStatus(videoId);
    }
  }, 1200);
}

/* --- Detail view --------------------------------------------------------- */

function renderDetail(video) {
  const result = video.result;
  const metadata = result.metadata || {};

  stopPlayback();
  // The poster is the same still the library shows, so the stage carries the
  // video's identity before a single frame has been decoded.
  player.poster = `/v1/videos/${video.video_id}/thumbnail`;
  player.src = `/v1/videos/${video.video_id}/source`;
  player.load();
  $("stage").style.setProperty("--ar",
    metadata.width && metadata.height ? `${metadata.width} / ${metadata.height}` : "16 / 9");
  $("stage-overlay").classList.remove("playing");
  $("t-dur").textContent = clock(result.duration);
  $("t-cur").textContent = "0:00.0";

  $("stage-badge").innerHTML = [
    metadata.width ? `${metadata.width}×${metadata.height}` : "",
    metadata.fps ? `${metadata.fps.toFixed(0)} fps` : "",
    clock(result.duration),
  ].filter(Boolean).map((text) => `<span>${esc(text)}</span>`).join("");

  renderInspector(video);
  renderTimeline(result);
  renderTranscript(result);
  renderScenes(video);
  renderHighlights(result);
  renderEdits(result);
  renderEntities(result);
  $("json").textContent = JSON.stringify(result, null, 2);

  $("c-transcript").textContent = result.transcript.length;
  $("c-scenes").textContent = result.scenes.length;
  $("c-highlights").textContent = result.highlights.length;
  $("c-edits").textContent = result.edit_suggestions.length;

  $("btn-download").disabled = !video.has_render;
  $("render-status").innerHTML = video.has_render
    ? '<span class="ok">A render from an earlier session is ready to download.</span>'
    : "";
  updateRenderPreview();
  resetDeleteButton();
  // Draw everything time-dependent once, so scene position and highlighting
  // are correct before the first frame plays.
  syncToTime();
}

function renderInspector(video) {
  const result = video.result;
  const metadata = result.metadata || {};

  $("v-title").textContent = result.title || video.filename;
  $("v-filename").textContent = video.filename;
  $("v-summary").textContent = result.summary || "No summary was produced for this video.";
  $("v-topics").innerHTML = (result.topics || [])
    .map((topic) => `<span class="tag">${esc(topic)}</span>`).join("");

  const language = result.transcript_language;
  const specs = [
    ["Duration", clock(result.duration)],
    ["Resolution", metadata.width ? `${metadata.width}×${metadata.height}` : "—"],
    ["Frame rate", metadata.fps ? `${metadata.fps.toFixed(2)} fps` : "—"],
    ["Size", bytes(metadata.size_bytes)],
    ["Codecs", [metadata.video_codec, metadata.audio_codec].filter(Boolean).join(" / ") || "—"],
    ["Audio", metadata.has_audio ? "yes" : "none"],
    ["Speech", language ? `${result.transcript.length} segments (${language})` : `${result.transcript.length} segments`],
    ["Uploaded", new Date(video.created_at).toLocaleString()],
    ["Video ID", video.video_id],
  ];
  $("v-specs").innerHTML = specs
    .map(([key, value]) => `<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`).join("");

  $("v-providers").innerHTML = Object.entries(result.providers || {}).length
    ? Object.entries(result.providers)
        .map(([key, value]) => `<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`).join("")
    : '<dt class="muted">—</dt><dd>not recorded</dd>';

  $("v-warnings").innerHTML = (result.warnings || [])
    .map((warning) => `<div class="warn-box">${esc(warning)}</div>`).join("");
}

/* --- Timeline ------------------------------------------------------------ */

const SCENE_COLORS = ["--scene-a", "--scene-b", "--scene-c", "--scene-d", "--scene-e", "--scene-f"];

function duration() {
  return (state.current && state.current.result && state.current.result.duration) || 0;
}

function pct(seconds) {
  const total = duration() || 1;
  return `${clamp((seconds / total) * 100, 0, 100)}%`;
}

function span(start, end) {
  const total = duration() || 1;
  const left = clamp((start / total) * 100, 0, 100);
  const width = clamp(((end - start) / total) * 100, 0.15, 100 - left);
  return { left: `${left}%`, width: `${width}%` };
}

function block(start, end, className, label, tooltip) {
  const node = el("div", `blk ${className || ""}`.trim());
  const geometry = span(start, end);
  node.style.left = geometry.left;
  node.style.width = geometry.width;
  node.dataset.start = start;
  node.dataset.end = end;
  if (tooltip) node.dataset.tip = tooltip;
  if (label) node.appendChild(el("span", null, esc(label)));
  return node;
}

function renderTimeline(result) {
  renderRuler(result.duration);
  renderFilmstrip();

  const scenes = $("track-scenes");
  scenes.innerHTML = "";
  result.scenes.forEach((scene, index) => {
    const color = `var(${SCENE_COLORS[index % SCENE_COLORS.length]})`;
    const node = block(scene.start, scene.end, "", `${index + 1}`,
      `${scene.description || "Scene " + (index + 1)}`);
    // Importance rides on opacity: a glance at the strip shows where the
    // model thought the video was carrying its weight.
    node.style.background = color;
    node.style.opacity = 0.45 + 0.55 * (scene.importance ?? 0.5);
    node.dataset.index = index;
    scenes.appendChild(node);
  });

  const speech = $("track-speech");
  speech.innerHTML = "";
  for (const segment of result.transcript) {
    speech.appendChild(block(segment.start, segment.end, "", "", segment.text));
  }

  const highlights = $("track-highlights");
  highlights.innerHTML = "";
  for (const highlight of result.highlights) {
    highlights.appendChild(block(highlight.start, highlight.end, "",
      highlight.title || "Highlight",
      `${highlight.title ? highlight.title + " — " : ""}${highlight.reason} (score ${highlight.score.toFixed(2)})`));
  }

  const edits = $("track-edits");
  edits.innerHTML = "";
  for (const edit of result.edit_suggestions) {
    edits.appendChild(block(edit.start, edit.end, edit.action, edit.action, edit.reason));
  }

  movePlayhead(0);
}

function renderRuler(total) {
  const ruler = $("ruler");
  ruler.innerHTML = "";
  if (!total) return;

  const width = $("timeline").clientWidth || 900;
  const targetTicks = clamp(Math.round(width / 90), 4, 40);
  const rawStep = total / targetTicks;
  const nice = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
  const step = nice.find((candidate) => candidate >= rawStep) || rawStep;

  for (let t = 0; t <= total + 0.001; t += step) {
    const tick = el("div", "tick");
    tick.style.left = pct(t);
    tick.appendChild(el("span", null, clock(t)));
    ruler.appendChild(tick);
  }
}

/** Evenly spaced stills across the video, so the strip reads like film. */
function renderFilmstrip() {
  const track = $("track-film");
  const total = duration();
  track.innerHTML = "";
  if (!total || !state.current) return;

  const count = clamp(Math.round(12 * state.zoom), 8, 60);
  const step = total / count;
  for (let i = 0; i < count; i += 1) {
    const start = i * step;
    const node = block(start, start + step, "", "", null);
    const img = el("img");
    img.loading = "lazy";
    img.alt = "";
    img.style.cssText = "width:100%;height:100%;object-fit:cover;display:block";
    img.src = `/v1/videos/${state.current.video_id}/frame?t=${(start + step / 2).toFixed(2)}&w=160`;
    img.addEventListener("error", () => { node.style.background = "var(--surface-3)"; img.remove(); });
    node.appendChild(img);
    track.appendChild(node);
  }
}

function movePlayhead(time) {
  $("playhead").style.left = pct(time);
}

/* Timeline interaction: click or drag anywhere scrubs; clicking a block jumps
   to where that block starts, which is what a cut list is for. */
(function bindTimeline() {
  const timeline = $("timeline");
  const tooltip = $("tl-tooltip");
  let scrubbing = false;

  const timeAt = (event) => {
    const rect = timeline.getBoundingClientRect();
    return clamp((event.clientX - rect.left) / rect.width, 0, 1) * duration();
  };

  timeline.addEventListener("pointerdown", (event) => {
    if (!duration()) return;
    const target = event.target.closest(".blk");
    const wasPlaying = !player.paused;
    seek(target ? parseFloat(target.dataset.start) : timeAt(event), { keepPlaying: wasPlaying });
    scrubbing = true;
    timeline.setPointerCapture(event.pointerId);
  });

  timeline.addEventListener("pointermove", (event) => {
    if (!duration()) return;
    const time = timeAt(event);
    $("hover-line").style.left = pct(time);
    if (scrubbing) seek(time, { keepPlaying: !player.paused });

    const target = event.target.closest(".blk");
    const text = target && target.dataset.tip;
    tooltip.classList.toggle("hidden", !text && !scrubbing);
    if (text || scrubbing) {
      const range = target
        ? `${clock(parseFloat(target.dataset.start))} – ${clock(parseFloat(target.dataset.end))}`
        : clockPrecise(time);
      tooltip.innerHTML = `<b>${esc(range)}</b>${text ? esc(text) : ""}`;
      const panel = document.querySelector(".timeline-panel").getBoundingClientRect();
      const left = clamp(event.clientX - panel.left - tooltip.offsetWidth / 2,
        8, panel.width - tooltip.offsetWidth - 8);
      tooltip.style.left = `${left}px`;
      tooltip.style.top = `${event.clientY - panel.top - tooltip.offsetHeight - 14}px`;
    }
  });

  const stop = () => { scrubbing = false; tooltip.classList.add("hidden"); };
  timeline.addEventListener("pointerup", stop);
  timeline.addEventListener("pointercancel", stop);
  timeline.addEventListener("mouseleave", stop);
  // Scrolling moves the timeline out from under a stationary pointer without
  // ever firing mouseleave, which would otherwise strand the tooltip on screen.
  document.querySelector(".detail-main").addEventListener("scroll", stop, { passive: true });
  $("timeline-scroll").addEventListener("scroll", stop, { passive: true });
})();

function setZoom(value) {
  state.zoom = clamp(value, 1, 8);
  $("timeline").style.setProperty("--zoom", state.zoom);
  $("zoom-level").textContent = `${state.zoom}×`;
  if (state.current && state.current.result) {
    renderRuler(state.current.result.duration);
    renderFilmstrip();
    movePlayhead(player.currentTime || 0);
  }
}

$("btn-zoom-in").addEventListener("click", () => setZoom(state.zoom * 2));
$("btn-zoom-out").addEventListener("click", () => setZoom(state.zoom / 2));

/* --- Tab content --------------------------------------------------------- */

function rowNode({ time, head, body, onClick }) {
  const node = el("button", "row-item");
  node.dataset.t = time;
  const header = el("div", "row-head", head);
  node.appendChild(header);
  if (body) node.appendChild(el("div", "body", body));
  node.addEventListener("click", onClick || (() => seek(time)));
  return node;
}

function renderTranscript(result) {
  const container = $("transcript");
  container.innerHTML = "";
  state.transcriptRows = [];
  $("transcript-lang").textContent = result.transcript_language
    ? `language: ${result.transcript_language}` : "";

  if (!result.transcript.length) {
    container.appendChild(el("div", "lib-empty", "No speech was detected in this video."));
    return;
  }

  result.transcript.forEach((segment) => {
    const node = rowNode({
      time: segment.start,
      head: `<span class="tc">${clock(segment.start)} – ${clock(segment.end)}</span>`,
      body: null,
    });
    const speech = el("div", "speech", esc(segment.text));
    node.appendChild(speech);
    node.dataset.end = segment.end;
    node.dataset.text = segment.text.toLowerCase();
    container.appendChild(node);
    state.transcriptRows.push(node);
  });
}

$("transcript-search").addEventListener("input", (event) => {
  const term = event.target.value.trim().toLowerCase();
  for (const row of state.transcriptRows) {
    const match = !term || row.dataset.text.includes(term);
    row.classList.toggle("hidden", !match);
    const speech = row.querySelector(".speech");
    const original = speech.textContent;
    if (!term) { speech.textContent = original; continue; }
    const index = original.toLowerCase().indexOf(term);
    speech.innerHTML = index < 0 ? esc(original)
      : `${esc(original.slice(0, index))}<mark>${esc(original.slice(index, index + term.length))}</mark>${esc(original.slice(index + term.length))}`;
  }
});

function renderScenes(video) {
  const result = video.result;
  const container = $("scenes");
  container.innerHTML = "";

  if (!result.scenes.length) {
    container.appendChild(el("div", "lib-empty", "No scenes were detected."));
    return;
  }

  result.scenes.forEach((scene, index) => {
    const card = el("button", "scene-card");
    card.dataset.index = index;
    card.dataset.t = scene.start;

    const shot = el("div", "scene-shot");
    const img = el("img");
    img.loading = "lazy";
    img.alt = "";
    img.src = `/v1/videos/${video.video_id}/frame?t=${((scene.start + scene.end) / 2).toFixed(2)}&w=320`;
    img.addEventListener("error", () => img.remove());
    shot.appendChild(img);
    shot.appendChild(el("span", "tc", `${clock(scene.start)} – ${clock(scene.end)}`));
    const importance = el("span", "imp", `imp ${(scene.importance ?? 0).toFixed(2)}`);
    importance.title = "How important the model judged this scene to be";
    shot.appendChild(importance);
    card.appendChild(shot);

    const body = el("div", "scene-body");
    body.appendChild(el("p", null, esc(scene.description || "No description.")));

    const tags = [...(scene.people || []), ...(scene.objects || []), ...(scene.actions || [])];
    if (tags.length) {
      body.appendChild(el("div", "chips",
        tags.slice(0, 8).map((tag) => `<span class="tag">${esc(tag)}</span>`).join("")));
    }
    if (scene.transcript_text) {
      body.appendChild(el("div", "quote", esc(`“${scene.transcript_text.slice(0, 170)}”`)));
    }
    card.appendChild(body);

    card.addEventListener("click", () => seek(scene.start));
    container.appendChild(card);
  });
}

function renderHighlights(result) {
  const container = $("highlights");
  container.innerHTML = "";
  if (!result.highlights.length) {
    container.appendChild(el("div", "lib-empty", "No highlights were identified."));
    return;
  }
  for (const highlight of result.highlights) {
    container.appendChild(rowNode({
      time: highlight.start,
      head: `<span class="tc">${clock(highlight.start)} – ${clock(highlight.end)}</span>
             <strong>${esc(highlight.title || "Highlight")}</strong>
             <span class="meter">score <i style="--v:${Math.round(highlight.score * 100)}%"></i>${highlight.score.toFixed(2)}</span>`,
      body: esc(highlight.reason),
    }));
  }
}

function renderEdits(result) {
  const container = $("edits");
  container.innerHTML = "";
  if (!result.edit_suggestions.length) {
    container.appendChild(el("div", "lib-empty", "No edit suggestions."));
    return;
  }
  for (const edit of result.edit_suggestions) {
    container.appendChild(rowNode({
      time: edit.start,
      head: `<span class="tc">${clock(edit.start)} – ${clock(edit.end)}</span>
             <span class="badge ${esc(edit.action)}">${esc(edit.action)}</span>
             <span class="meter">confidence <i style="--v:${Math.round((edit.confidence ?? 0.5) * 100)}%"></i>${(edit.confidence ?? 0.5).toFixed(2)}</span>`,
      body: esc(edit.reason),
    }));
  }
}

function renderEntities(result) {
  const container = $("entities");
  const groups = [
    ["People", result.people],
    ["Objects", result.objects],
    ["Actions", result.actions],
    ["Topics", result.topics],
  ].filter(([, values]) => values && values.length);

  container.innerHTML = groups.length
    ? groups.map(([label, values]) => `
        <div class="entity-group">
          <h4>${esc(label)} <span class="muted">(${values.length})</span></h4>
          <div class="chips">${values.map((v) => `<span class="tag">${esc(v)}</span>`).join("")}</div>
        </div>`).join("")
    : '<div class="lib-empty">Nothing was extracted for this video.</div>';
}

/* --- Tabs ---------------------------------------------------------------- */

$("tabs").addEventListener("click", (event) => {
  const tab = event.target.closest(".tab");
  if (!tab) return;
  for (const node of document.querySelectorAll(".tab")) {
    const active = node === tab;
    node.classList.toggle("active", active);
    node.setAttribute("aria-selected", String(active));
  }
  for (const pane of document.querySelectorAll(".tab-pane")) {
    pane.classList.toggle("active", pane.id === `pane-${tab.dataset.tab}`);
  }
  $("follow-wrap").classList.toggle("hidden", tab.dataset.tab !== "transcript");
});

$("opt-follow").addEventListener("change", (event) => {
  state.followPlayback = event.target.checked;
});

/* --- Playback ------------------------------------------------------------ */

function seek(time, { keepPlaying = true } = {}) {
  if (!isFinite(time)) return;
  player.currentTime = clamp(time, 0, duration() || player.duration || 0);
  syncToTime();
  if (keepPlaying && player.paused && player.currentTime > 0 && state.userHasPlayed) {
    player.play().catch(() => {});
  }
}

function stopPlayback() {
  try { player.pause(); } catch { /* nothing loaded */ }
  document.body.classList.remove("is-playing");
  state.userHasPlayed = false;
  state.activeSceneIndex = -1;
}

function togglePlay() {
  if (player.paused) {
    state.userHasPlayed = true;
    player.play().catch((error) => toast(`Playback failed: ${error.message}`, "err"));
  } else {
    player.pause();
  }
}

player.addEventListener("play", () => {
  document.body.classList.add("is-playing");
  $("stage-overlay").classList.add("playing");
  state.userHasPlayed = true;
});
player.addEventListener("pause", () => {
  document.body.classList.remove("is-playing");
  $("stage-overlay").classList.remove("playing");
});
player.addEventListener("timeupdate", syncToTime);
player.addEventListener("seeked", syncToTime);
player.addEventListener("volumechange", () => {
  document.body.classList.toggle("is-muted", player.muted || player.volume === 0);
});
player.addEventListener("error", () => {
  if (player.src) toast("The browser could not play this file. The analysis is still available.", "err", 6000);
});

$("btn-play").addEventListener("click", togglePlay);
$("big-play").addEventListener("click", togglePlay);
$("stage").addEventListener("click", (event) => {
  if (event.target === player) togglePlay();
});
player.addEventListener("dblclick", toggleFullscreen);
$("btn-back").addEventListener("click", () => seek(player.currentTime - 5));
$("btn-fwd").addEventListener("click", () => seek(player.currentTime + 5));
$("btn-prev-scene").addEventListener("click", () => jumpScene(-1));
$("btn-next-scene").addEventListener("click", () => jumpScene(1));
$("speed").addEventListener("change", (event) => { player.playbackRate = parseFloat(event.target.value); });
$("btn-mute").addEventListener("click", () => { player.muted = !player.muted; });
$("volume").addEventListener("input", (event) => {
  player.volume = parseFloat(event.target.value);
  player.muted = player.volume === 0;
});
$("btn-captions").addEventListener("click", toggleCaptions);
$("btn-full").addEventListener("click", toggleFullscreen);

function toggleCaptions() {
  state.captionsOn = !state.captionsOn;
  $("btn-captions").classList.toggle("on", state.captionsOn);
  $("stage-caption").classList.toggle("hidden", !state.captionsOn);
  syncToTime();
}

function toggleFullscreen() {
  const stage = $("stage");
  if (document.fullscreenElement) document.exitFullscreen();
  else stage.requestFullscreen?.().catch(() => {});
}

function jumpScene(direction) {
  const scenes = state.current?.result?.scenes || [];
  if (!scenes.length) return;
  const time = player.currentTime + 0.25 * direction;
  let index = scenes.findIndex((scene) => time >= scene.start && time < scene.end);
  if (index < 0) index = direction > 0 ? -1 : scenes.length;
  const next = clamp(index + direction, 0, scenes.length - 1);
  seek(scenes[next].start);
}

/** One function drives everything that depends on the current time. */
function syncToTime() {
  const time = player.currentTime || 0;
  $("t-cur").textContent = clockPrecise(time);
  movePlayhead(time);

  const result = state.current?.result;
  if (!result) return;

  // Scenes: timeline block, scene card and the position readout.
  const sceneIndex = result.scenes.findIndex((scene) => time >= scene.start && time < scene.end);
  if (sceneIndex !== state.activeSceneIndex) {
    state.activeSceneIndex = sceneIndex;
    for (const node of document.querySelectorAll("#track-scenes .blk")) {
      node.classList.toggle("current", Number(node.dataset.index) === sceneIndex);
    }
    for (const card of document.querySelectorAll(".scene-card")) {
      card.classList.toggle("current", Number(card.dataset.index) === sceneIndex);
    }
    $("scene-pos").textContent = sceneIndex >= 0
      ? `Scene ${sceneIndex + 1} / ${result.scenes.length}`
      : `— / ${result.scenes.length}`;
  }

  // Transcript: highlight the line being spoken, and optionally follow it.
  let active = null;
  for (const row of state.transcriptRows) {
    const start = parseFloat(row.dataset.t);
    const end = parseFloat(row.dataset.end);
    const isCurrent = time >= start && time < end;
    if (isCurrent && !row.classList.contains("current")) active = row;
    row.classList.toggle("current", isCurrent);
  }
  if (active && state.followPlayback && !player.paused) {
    const container = $("transcript");
    const offset = active.offsetTop - container.clientHeight / 2 + active.clientHeight / 2;
    container.scrollTo({ top: Math.max(0, offset), behavior: "smooth" });
  }

  if (state.captionsOn) {
    const line = result.transcript.find((segment) => time >= segment.start && time < segment.end);
    const caption = $("stage-caption");
    caption.textContent = line ? line.text : "";
    caption.style.visibility = line ? "visible" : "hidden";
  }
}

/* --- Export -------------------------------------------------------------- */

for (const id of ["opt-remove", "opt-captions", "opt-highlights"]) {
  $(id).addEventListener("change", updateRenderPreview);
}

/* The next two mirror `rendering.merge_ranges` / `rendering.subtract_ranges`.
   Duplicating them buys an exact preview: the two options are not exclusive on
   the server — a highlight reel still has its removals cut out of it — and a
   preview that guessed otherwise would contradict the render it describes. */

function mergeRanges(ranges, tolerance = 0.05) {
  if (!ranges.length) return [];
  const ordered = ranges
    .map(([a, b]) => [Math.min(a, b), Math.max(a, b)])
    .sort((x, y) => x[0] - y[0]);

  const merged = [ordered[0].slice()];
  for (const [start, end] of ordered.slice(1)) {
    const last = merged[merged.length - 1];
    if (start <= last[1] + tolerance) last[1] = Math.max(last[1], end);
    else merged.push([start, end]);
  }
  return merged;
}

function subtractRanges(base, holes) {
  let remaining = base;
  for (const [holeStart, holeEnd] of mergeRanges(holes)) {
    const next = [];
    for (const [start, end] of remaining) {
      if (holeEnd <= start || holeStart >= end) { next.push([start, end]); continue; }
      if (holeStart > start) next.push([start, holeStart]);
      if (holeEnd < end) next.push([holeEnd, end]);
    }
    remaining = next;
  }
  return remaining.filter(([start, end]) => end - start > 0.05);
}

function updateRenderPreview() {
  const result = state.current?.result;
  const node = $("render-preview");
  if (!result) { node.textContent = ""; return; }

  const total = result.duration;
  const highlightsOnly = $("opt-highlights").checked;
  const notes = [];

  let keep = [[0, total]];
  if (highlightsOnly) {
    const windows = (result.highlights || [])
      .filter((h) => h.end > h.start)
      .map((h) => [h.start, Math.min(h.end, total)]);
    if (windows.length) {
      keep = mergeRanges(windows);
      notes.push(`${windows.length} highlight${windows.length === 1 ? "" : "s"}`);
    } else {
      notes.push("no highlights to reel — keeping everything");
    }
  }

  if ($("opt-remove").checked) {
    const removals = (result.edit_suggestions || [])
      .filter((edit) => edit.action === "remove" && edit.end > edit.start)
      .map((edit) => [edit.start, Math.min(edit.end, total)]);
    if (removals.length) {
      keep = subtractRanges(keep, removals);
      notes.push(`${removals.length} removal${removals.length === 1 ? "" : "s"} applied`);
    }
  }

  const kept = keep.reduce((sum, [start, end]) => sum + (end - start), 0);
  if (!notes.length) notes.push("no cuts — a re-encode of the original");
  if ($("opt-captions").checked) notes.push("captions burned in");

  const empty = kept <= 0;
  node.innerHTML = empty
    ? '<span class="err">This combination removes the whole video.</span>'
    // Tenths, because a cut of half a second is a real difference at this scale.
    : `<b>${clockPrecise(total)}</b> → <b>${clockPrecise(kept)}</b> · ${esc(notes.join(" · "))}` +
      (keep.length > 1 ? ` · ${keep.length} pieces` : "");
  $("btn-render").disabled = empty;
}

$("btn-render").addEventListener("click", async () => {
  const video = state.current;
  if (!video) return;
  const button = $("btn-render");
  button.disabled = true;
  $("render-status").innerHTML = '<span class="muted">Rendering with FFmpeg…</span>';

  try {
    const data = await api(`/v1/videos/${video.video_id}/render`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        apply_removals: $("opt-remove").checked,
        burn_captions: $("opt-captions").checked,
        highlights_only: $("opt-highlights").checked,
      }),
    });
    $("render-status").innerHTML =
      `<span class="ok">Rendered.</span> ${data.source_duration.toFixed(1)}s → ` +
      `${data.output_duration.toFixed(1)}s across ${data.segments_kept} segment(s)` +
      (data.segments_removed ? `, ${data.segments_removed} removed` : "") + ".";
    $("btn-download").disabled = false;
    state.current.has_render = true;
    toast("Edit rendered", "ok");
  } catch (error) {
    $("render-status").innerHTML = `<span class="err">${esc(error.message)}</span>`;
    toast(`Render failed: ${error.message}`, "err", 7000);
  } finally {
    button.disabled = false;
  }
});

$("btn-download").addEventListener("click", () => {
  if (state.current) window.location = `/v1/videos/${state.current.video_id}/render/download`;
});
$("btn-srt").addEventListener("click", () => {
  if (state.current) window.location = `/v1/videos/${state.current.video_id}/transcript.srt`;
});
$("btn-copy-transcript").addEventListener("click", async () => {
  const text = (state.current?.result?.transcript || [])
    .map((segment) => `[${clock(segment.start)}] ${segment.text}`).join("\n");
  await copy(text, $("btn-copy-transcript"), "Copy text");
});
$("btn-copy").addEventListener("click", async () => {
  await copy(JSON.stringify(state.current?.result ?? {}, null, 2), $("btn-copy"), "Copy JSON");
});
$("btn-json").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(state.current?.result ?? {}, null, 2)], { type: "application/json" });
  const link = el("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${state.current?.video_id || "video"}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
});

async function copy(text, button, label) {
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = "Copied";
    setTimeout(() => { button.textContent = label; }, 1400);
  } catch {
    toast("The browser blocked clipboard access", "err");
  }
}

/* Delete arms itself on the first click rather than opening a modal dialog:
   one fewer interruption, and the destructive step still takes two actions. */
function resetDeleteButton() {
  state.deleteArmed = false;
  const button = $("btn-delete");
  button.classList.remove("armed");
  button.title = "Delete this video";
}

$("btn-delete").addEventListener("click", async () => {
  const video = state.current;
  if (!video) return;

  if (!state.deleteArmed) {
    state.deleteArmed = true;
    $("btn-delete").classList.add("armed");
    $("btn-delete").title = "Click again to delete permanently";
    toast("Click delete again to remove this video and its files", "info", 4000);
    setTimeout(resetDeleteButton, 4000);
    return;
  }

  try {
    await api(`/v1/videos/${video.video_id}`, { method: "DELETE" });
    toast("Video deleted", "ok");
    location.hash = "";
    state.library.items = state.library.items.filter((item) => item.video_id !== video.video_id);
    await loadLibrary();
  } catch (error) {
    toast(`Could not delete: ${error.message}`, "err");
  }
});

/* --- Shortcuts ----------------------------------------------------------- */

const modal = $("modal-shortcuts");
$("btn-shortcuts").addEventListener("click", () => modal.classList.remove("hidden"));
$("btn-close-shortcuts").addEventListener("click", () => modal.classList.add("hidden"));
modal.addEventListener("click", (event) => {
  if (event.target === modal) modal.classList.add("hidden");
});

window.addEventListener("keydown", (event) => {
  const tag = (event.target.tagName || "").toLowerCase();
  const typing = tag === "input" || tag === "textarea" || tag === "select" || event.target.isContentEditable;

  if (event.key === "Escape") {
    modal.classList.add("hidden");
    if (typing) event.target.blur();
    return;
  }
  if (event.key === "/" && !typing) {
    event.preventDefault();
    $("search").focus();
    $("search").select();
    return;
  }
  if (event.key === "?" && !typing) {
    modal.classList.toggle("hidden");
    return;
  }
  if (typing || event.metaKey || event.ctrlKey || event.altKey) return;
  if (!state.current || state.current.status !== "completed") return;

  const fps = state.current.result?.metadata?.fps || 30;
  const actions = {
    " ": () => togglePlay(),
    k: () => togglePlay(),
    j: () => seek(player.currentTime - 10),
    l: () => seek(player.currentTime + 10),
    ArrowLeft: () => seek(player.currentTime - (event.shiftKey ? 1 : 5)),
    ArrowRight: () => seek(player.currentTime + (event.shiftKey ? 1 : 5)),
    ",": () => { player.pause(); seek(player.currentTime - 1 / fps, { keepPlaying: false }); },
    ".": () => { player.pause(); seek(player.currentTime + 1 / fps, { keepPlaying: false }); },
    "[": () => jumpScene(-1),
    "]": () => jumpScene(1),
    m: () => { player.muted = !player.muted; },
    c: () => toggleCaptions(),
    f: () => toggleFullscreen(),
  };

  const action = actions[event.key] || actions[event.key.toLowerCase()];
  if (action) { event.preventDefault(); action(); return; }

  if (/^[0-9]$/.test(event.key)) {
    event.preventDefault();
    seek((duration() * Number(event.key)) / 10);
  }
});

/* Re-drawing the ruler on resize keeps tick spacing readable at any width. */
window.addEventListener("resize", debounce(() => {
  if (state.current?.result) {
    renderRuler(state.current.result.duration);
    movePlayhead(player.currentTime || 0);
  }
}, 180));

/* --- Boot ---------------------------------------------------------------- */

loadHealth();
setInterval(loadHealth, 30000);
loadLibrary().then(route);
