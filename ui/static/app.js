// Adversary UI — vanilla JS. No build step.

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
};

const state = {
  chatHistory: [],
  songsCount: 0,
  ready: false,
  model: "",
  baseUrl: "",
  apiKeySet: false,
  serverUp: false,
  online: false,
  antakshari: null,
  timeBias: 0.0,
  installed: [],
  chatMode: "lyric",
};

// --------------------------------------------------------------------------
// API helpers
// --------------------------------------------------------------------------

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return r;
}

// Read SSE from a fetch POST response. EventSource only supports GET, so we
// parse the text stream manually: blank-line-delimited frames, each with
// `event: <name>` and `data: <json>` lines.
async function readSSE(response, { onProgress, onDone, onError }) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      let evName = "message";
      let dataLine = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) evName = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLine = line.slice(5).trim();
      }
      let data = {};
      if (dataLine) {
        try { data = JSON.parse(dataLine); } catch (e) { data = { raw: dataLine }; }
      }
      if (evName === "progress" && onProgress) onProgress(data);
      else if (evName === "done" && onDone) { onDone(data); return; }
      else if (evName === "error" && onError) { onError(data); return; }
    }
  }
}

// --------------------------------------------------------------------------
// Rendering
// --------------------------------------------------------------------------

function setProgress(node, frac, label) {
  node.classList.remove("hidden");
  const fill = node.querySelector(".fill");
  const lbl = node.querySelector(".label");
  if (frac != null) fill.style.width = `${Math.round(frac * 100)}%`;
  if (label != null) lbl.textContent = label;
}

function renderSidebarStatus() {
  $("#collection-status").textContent =
    `Status: ${state.ready ? "ready" : "no index"} · ${state.songsCount} songs`;

  const statusEl = $("#model-status");
  statusEl.innerHTML = "";
  const dot = el("span", "dot");
  dot.classList.toggle("up", state.serverUp || state.online);
  statusEl.appendChild(dot);
  let statusText;
  if (state.online) {
    statusText = state.serverUp ? "Online endpoint" : "Online endpoint (probe failed)";
  } else {
    statusText = state.serverUp ? "Ollama: online" : "Ollama: offline";
  }
  statusEl.append(` ${statusText}  ${state.baseUrl}`);
  $("#model-active").textContent = `Active model: ${state.model}`;

  $("#time-bias").value = String(state.timeBias);
  $("#time-bias-value").textContent = Number(state.timeBias).toFixed(1);

  const note = $("#cloud-note");
  if (state.online) {
    note.textContent = "Cloud mode active (API key set or base URL is non-local) — models are called via API key, never pulled.";
  } else {
    note.textContent = "Cloud Model inactive — no API key set and base URL is local. Set an API key (and point OLLAMA_BASE_URL at the cloud host) to enable.";
  }
  if (state.apiKeySet) {
    $("#api-key").placeholder = "•••••• (set)";
    $("#api-key").value = "";
  } else {
    $("#api-key").placeholder = "Ollama API key";
  }

  // Global status chips in the top bar.
  const gs = $("#global-status");
  gs.innerHTML = "";
  const idxChip = el("span", "chip", state.ready ? `${state.songsCount} songs` : "no index");
  const modelChip = el("span", "chip", state.model || "no model");
  const modeChip = el("span", "chip", state.chatMode === "prose" ? "prose" : "lyric");
  gs.appendChild(idxChip);
  gs.appendChild(modelChip);
  gs.appendChild(modeChip);
}

function renderAntakshari() {
  const panel = $("#antakshari-panel");
  const body = $("#antakshari-body");
  body.innerHTML = "";
  if (!state.antakshari) {
    panel.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");
  const s = state.antakshari;
  if (s.current_song) {
    body.appendChild(el("p", null,
      `AI's last song: ${s.current_song.artist} — ${s.current_song.title}`));
  }
  body.appendChild(el("p", null, `Required letter: ${s.required_character}`));
  body.appendChild(el("p", null, `Score: ${s.score}`));
  body.appendChild(el("p", "caption",
    `Used songs: ${s.used_songs_count} · Used artists: ${s.used_artists_count}`));
}

function renderSongRow(s) {
  const row = el("div", "song-row");
  row.appendChild(el("div", "title", s.title));
  row.appendChild(el("div", "artist", s.artist));
  const meta = el("div", "meta");
  if (s.provider) meta.appendChild(el("span", "pill", s.provider));
  if (s.duration) meta.appendChild(el("span", "pill", s.duration));
  if (s.tags && s.tags.length) {
    for (const t of s.tags) meta.appendChild(el("span", "pill", t));
  }
  row.appendChild(meta);
  return row;
}

function renderSongsList(songs) {
  const list = $("#songs-list");
  list.innerHTML = "";
  $("#songs-count").textContent = `${songs.length} songs in index`;
  if (!songs.length) {
    list.appendChild(el("p", "caption", "No index yet — build one from the left panel."));
    return;
  }
  for (const s of songs) list.appendChild(renderSongRow(s));
}

function escapeHtml(s) {
  return (s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderMarkdown(text) {
  if (!text) return "";
  // Escape HTML first (safety for external lyric content).
  let html = escapeHtml(text);
  // Bold: **text** → <strong>text</strong>
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  // Italic: *text* → <em>text</em> (but not inside bold)
  html = html.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, "<em>$1</em>");

  const lines = html.split("\n");
  let out = [];
  let inList = false;
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("- ")) {
      if (!inList) { out.push('<ul class="md-list">'); inList = true; }
      out.push(`<li>${trimmed.slice(2)}</li>`);
    } else {
      if (inList) { out.push("</ul>"); inList = false; }
      if (trimmed === "") {
        out.push('<div class="md-gap"></div>');
      } else if (trimmed.startsWith("(") && trimmed.endsWith(")")) {
        out.push(`<p class="md-provenance">${trimmed}</p>`);
      } else {
        out.push(`<p class="md-p">${trimmed}</p>`);
      }
    }
  }
  if (inList) out.push("</ul>");
  return out.join("");
}

function renderTurn(turn) {
  const turn_el = el("div", "turn");

  const user_b = el("div", "bubble user", turn.user);
  turn_el.appendChild(user_b);

  const a_b = el("div", "bubble assistant");
  if (turn.mode && turn.mode !== "general") {
    const tag = el("span", "mode-tag", turn.mode);
    a_b.appendChild(tag);
  }
  const md = el("div", "md-body");
  md.innerHTML = renderMarkdown(turn.assistant || "");
  a_b.appendChild(md);
  turn_el.appendChild(a_b);

  if (turn.rewritten_query) {
    const rq = el("div", "caption", `Retrieval query: ${turn.rewritten_query}`);
    turn_el.appendChild(rq);
  }

  const sources = turn.sources || [];
  const lyricData = turn.lyric_data || {};
  if (sources.length) {
    const displays = sources.map((s) => s.display || "").filter(Boolean).join(" | ");
    const sc = el("div", "sources", `Sources: ${displays}`);
    turn_el.appendChild(sc);
    for (const src of sources) {
      turn_el.appendChild(renderSourceExtras(src, lyricData));
    }
  }
  return turn_el;
}

function renderSourceExtras(src, lyricData) {
  const row = el("div", "source-row");
  if (!src || !src.track_id) return row;
  row.dataset.trackId = src.track_id;

  // We need provider / preview / tags — look them up from the cached songs.
  const song = (state._songsCache || []).find((s) => s.track_id === src.track_id);
  const provider = song ? song.provider : "";
  const preview = song ? song.preview_url : "";
  const tags = song ? song.tags : [];

  row.appendChild(el("div", "src-meta", `Provider: ${provider}`));
  if (tags && tags.length) {
    row.appendChild(el("div", "src-meta", `Tags: ${tags.join(", ")}`));
  }
  if (preview) {
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.src = preview;
    row.appendChild(audio);
  }
  const stored = lyricData && lyricData[src.track_id];
  const btn = el("button", "lyrics-btn", stored ? "Lyrics" : "Lyrics");
  btn.addEventListener("click", () => {
    if (stored) {
      showStoredLyrics(stored, row);
    } else {
      fetchLyrics(src.track_id, row);
    }
  });
  row.appendChild(btn);
  return row;
}

function showStoredLyrics(stored, row) {
  let out = row.querySelector(".lyrics-out");
  if (!out) {
    out = el("div", "lyrics-out");
    row.appendChild(out);
  }
  out.innerHTML = "";

  // --- Song details ---
  const details = el("div", "song-details");
  if (stored.title || stored.artist) {
    details.appendChild(el("div", "song-details-title",
      `${stored.artist || ""} — ${stored.title || ""}`));
  }
  const meta = el("div", "song-details-meta");
  if (stored.album) meta.appendChild(el("span", null, `Album: ${stored.album}`));
  if (stored.release_date) meta.appendChild(el("span", null, `Released: ${stored.release_date}`));
  if (stored.source) meta.appendChild(el("span", null, `Provider: ${stored.source}`));
  if (stored.duration) {
    const m = Math.floor(stored.duration / 60);
    const s = String(stored.duration % 60).padStart(2, "0");
    meta.appendChild(el("span", null, `Duration: ${m}:${s}`));
  }
  if (meta.children.length) details.appendChild(meta);
  if (stored.tags && stored.tags.length) {
    const tagLine = el("div", "song-details-tags");
    for (const t of stored.tags) tagLine.appendChild(el("span", "pill", t));
    details.appendChild(tagLine);
  }
  if (details.children.length) out.appendChild(details);

  // --- Lyrics ---
  const phrases = stored.phrases || [];
  if (phrases.length) {
    const heading = el("div", "lyrics-heading", "Lyrics used");
    out.appendChild(heading);
    for (const p of phrases) {
      const line = el("div", "lyric-line", `\u201c${p}\u201d`);
      out.appendChild(line);
    }
  } else {
    out.appendChild(el("div", "err", "No lyric phrases were stored for this source."));
  }
  if (stored.provider) {
    out.appendChild(el("div", "caption", `(Lyrics source: ${stored.provider})`));
  }
  if (stored.is_synced) {
    out.appendChild(el("div", "caption", "(Lyrics type: synced — timing tags stripped)"));
  }
  if (stored.romanized) {
    out.appendChild(el("div", "caption", "(Lyrics transliterated to Latin)"));
  }
}

function renderMessages() {
  const msgs = $("#messages");
  msgs.innerHTML = "";
  for (const t of state.chatHistory) msgs.appendChild(renderTurn(t));
  const scroll = $("#chat-scroll");
  scroll.scrollTop = scroll.scrollHeight;
}

function renderAll() {
  renderSidebarStatus();
  renderAntakshari();
  renderMessages();
  applyChatInputMode();
}

function applyChatInputMode() {
  const input = $("#chat-input");
  const antakActive = !!(state.antakshari && state.antakshari.active);
  if (antakActive) {
    // Antakshari takes precedence — accept full song titles.
    input.maxLength = 524288;
    input.placeholder = "Type a song title from your collection…";
    input.removeAttribute("pattern");
  } else if (state.chatMode === "prose") {
    // Lyric Prose Recommendation: only a single letter.
    input.maxLength = 1;
    input.placeholder = "Enter a single letter (A–Z)…";
  } else {
    // Lyrical conversational: normal free-text chat.
    input.maxLength = 524288;
    input.placeholder = "Ask for a song, a mood, a theme…";
  }
}

// --------------------------------------------------------------------------
// State fetch
// --------------------------------------------------------------------------

async function fetchState() {
  try {
    const s = await getJSON("/api/state");
    state.chatHistory = s.chat_history || [];
    state.songsCount = s.songs_count || 0;
    state.ready = !!s.ready;
    state.model = s.model || "";
    state.baseUrl = s.base_url || "";
    state.apiKeySet = !!s.api_key_set;
    state.serverUp = !!s.server_up;
    state.online = !!s.online;
    state.antakshari = s.antakshari || null;
    state.timeBias = s.time_bias || 0.0;
    renderAll();
  } catch (e) {
    console.error("fetchState failed", e);
  }
  try {
    const songs = await getJSON("/api/songs");
    state._songsCache = songs;
    renderSongsList(songs);
  } catch (e) {
    console.error("fetchSongs failed", e);
  }
  try {
    const m = await getJSON("/api/models");
    state.installed = m.installed || [];
    renderInstalledModels(m);
  } catch (e) {
    console.error("fetchModels failed", e);
  }
}

function renderInstalledModels(m) {
  const sel = $("#installed-select");
  sel.innerHTML = "";
  const installed = m.installed || [];
  const OTHER = "✏️ Other (type a name)…";
  if (installed.length === 0) {
    const opt = document.createElement("option");
    opt.value = OTHER; opt.textContent = OTHER;
    sel.appendChild(opt);
  } else {
    for (const name of installed) {
      const opt = document.createElement("option");
      opt.value = name; opt.textContent = name;
      sel.appendChild(opt);
    }
    const opt = document.createElement("option");
    opt.value = OTHER; opt.textContent = OTHER;
    sel.appendChild(opt);
  }
  // Pre-select the configured model if present.
  let chosen = installed.includes(m.configured) ? m.configured : installed[0] || OTHER;
  sel.value = chosen;
  toggleLocalModelInput(sel.value === OTHER);
  $("#installed-list").textContent = installed.length
    ? "Installed: " + installed.join("  ·  ")
    : "No models installed locally.";
}

function toggleLocalModelInput(show) {
  $("#local-model").classList.toggle("hidden", !show);
}

// --------------------------------------------------------------------------
// Actions
// --------------------------------------------------------------------------

async function sendMessage() {
  const input = $("#chat-input");
  const msg = input.value.trim();
  if (!msg) return;
  if (!state.ready) {
    alert("Build the index first (left panel).");
    return;
  }
  // In Lyric Prose Recommendation mode, only a single letter is accepted.
  const antakActive = !!(state.antakshari && state.antakshari.active);
  if (state.chatMode === "prose" && !antakActive) {
    const letter = (msg.match(/[A-Za-z]/) || [""])[0];
    if (!letter) {
      alert("Please enter a single letter (A–Z).");
      return;
    }
  }
  input.value = "";
  const sendBtn = $("#send-btn");
  sendBtn.disabled = true;

  // Show the user's message + a spinner immediately.
  const msgs = $("#messages");
  const userTurn = el("div", "turn");
  userTurn.appendChild(el("div", "bubble user", msg));
  const spinBubble = el("div", "spinner-bubble");
  spinBubble.appendChild(el("div", "spinner"));
  spinBubble.appendChild(el("div", "spinner-label", "Thinking…"));
  userTurn.appendChild(spinBubble);
  msgs.appendChild(userTurn);
  const scroll = $("#chat-scroll");
  scroll.scrollTop = scroll.scrollHeight;

  const override = state.chatMode; // "lyric" or "prose"
  try {
    const r = await postJSON("/api/chat", { message: msg, override });
    if (!r.ok) {
      const err = await r.json().catch(() => ({ error: r.statusText }));
      spinBubble.querySelector(".spinner-label").textContent = `Error: ${err.error || r.statusText}`;
      spinBubble.querySelector(".spinner").style.display = "none";
      return;
    }
    const data = await r.json();
    if (data.error) {
      spinBubble.querySelector(".spinner-label").textContent = `Error: ${data.error}`;
      spinBubble.querySelector(".spinner").style.display = "none";
      return;
    }

    // Remove the spinner turn and append the real turn via renderAll.
    msgs.removeChild(userTurn);
    if (data.turn) state.chatHistory.push(data.turn);
    if (data.antakshari !== undefined) state.antakshari = data.antakshari;
    renderAll();
    // Refresh songs cache in case ingestion happened elsewhere.
    fetchSongs();
  } catch (e) {
    spinBubble.querySelector(".spinner-label").textContent = `Error: ${e.message}`;
    spinBubble.querySelector(".spinner").style.display = "none";
  } finally {
    sendBtn.disabled = false;
  }
}

async function fetchSongs() {
  try {
    const songs = await getJSON("/api/songs");
    state._songsCache = songs;
    state.songsCount = songs.length;
    renderSongsList(songs);
    renderSidebarStatus();
  } catch (e) { /* ignore */ }
}

async function ingestYouTube() {
  const url = $("#youtube-url").value.trim();
  if (!url) { alert("Enter a YouTube playlist URL or ID first."); return; }
  const enrich = $("#enrich-toggle").checked;
  const prog = $("#ingest-progress");
  const btn = $("#build-btn");
  btn.disabled = true;
  setProgress(prog, 0.0, "Fetching playlist …");
  const r = await postJSON("/api/ingest/youtube", { url, enrich });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ error: r.statusText }));
    btn.disabled = false;
    prog.classList.add("hidden");
    alert(`YouTube import failed: ${err.error || r.statusText}`);
    return;
  }
  await readSSE(r, {
    onProgress: (d) => setProgress(prog, d.frac, d.label),
    onDone: async (d) => {
      setProgress(prog, 1.0, `Indexed ${d.n || ""} songs from YouTube.`);
      btn.disabled = false;
      await fetchState();
      setTimeout(() => prog.classList.add("hidden"), 1500);
    },
    onError: (d) => {
      prog.classList.add("hidden");
      btn.disabled = false;
      alert(`YouTube import failed: ${d.error || "unknown error"}`);
    },
  });
}

async function pullModel(name) {
  const prog = $("#pull-progress");
  setProgress(prog, 0.0, `Pulling ${name} … (needs network)`);
  const r = await postJSON("/api/pull", { model: name });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ error: r.statusText }));
    prog.classList.add("hidden");
    alert(`Pull failed: ${err.error || r.statusText}`);
    return false;
  }
  let ok = false;
  await readSSE(r, {
    onProgress: (d) => setProgress(prog, d.frac, `${d.label || "pulling"} (${Math.round((d.frac || 0) * 100)}%)`),
    onDone: async () => {
      ok = true;
      setProgress(prog, 1.0, `Done. ${name} is installed.`);
      await switchModel(name);
      setTimeout(() => prog.classList.add("hidden"), 1500);
    },
    onError: (d) => {
      prog.classList.add("hidden");
      alert(`Pull failed: ${d.error || "unknown error"}`);
    },
  });
  return ok;
}

async function switchModel(name) {
  const r = await postJSON("/api/model", { model: name });
  const data = await r.json();
  if (r.ok && data.ok) {
    await fetchState();
    return true;
  }
  if (data.error === "not_installed") {
    // Pull, then switch.
    return await pullModel(name);
  }
  alert(`Error: ${data.error || r.statusText}`);
  return false;
}

async function setApiKey() {
  const k = $("#api-key").value.trim();
  await postJSON("/api/api_key", { api_key: k });
  $("#api-key").value = "";
  await fetchState();
}

async function clearChatHistory() {
  await postJSON("/api/clear/chat", {});
  state.chatHistory = [];
  state.antakshari = null;
  await fetchState();
}

async function clearCollection() {
  const r = await postJSON("/api/clear/collection", {});
  const data = await r.json().catch(() => ({}));
  state.antakshari = null;
  await fetchState();
  if (data.removed_embeddings) {
    console.log(`Cleared ${data.removed_embeddings} cached embeddings.`);
  }
}

async function clearApiKey() {
  await postJSON("/api/clear/api_key", {});
  await fetchState();
}

async function setTimeBias(v) {
  await postJSON("/api/time_bias", { bias: parseFloat(v) });
  state.timeBias = parseFloat(v);
  $("#time-bias-value").textContent = state.timeBias.toFixed(1);
}

async function stopAntakshari() {
  const r = await postJSON("/api/antakshari/stop", {});
  const data = await r.json();
  state.antakshari = data.session || null;
  renderAntakshari();
  await fetchState();
}

async function fetchLyrics(trackId, row) {
  let out = row.querySelector(".lyrics-out");
  if (!out) {
    out = el("div", "lyrics-out");
    row.appendChild(out);
  }
  out.textContent = "Fetching lyrics …";
  try {
    const res = await getJSON(`/api/lyrics/${encodeURIComponent(trackId)}`);
    if (res.available) {
      const lines = [];
      lines.push(res.excerpt || "");
      if (res.provider) lines.push(`(Lyrics source: ${res.provider})`);
      if (res.is_synced) lines.push("(Lyrics type: synced — timing tags stripped)");
      if (res.romanized) lines.push("(Lyrics transliterated to Latin)");
      out.textContent = lines.filter(Boolean).join("\n");
    } else {
      out.innerHTML = "";
      const e = el("div", "err", "Lyrics not available.");
      const hint = el("div", "caption",
        "Try switching providers (LYRICS_PROVIDER=musixmatch with a user token has better Hindi/Urdu coverage).");
      out.appendChild(e);
      out.appendChild(hint);
    }
  } catch (e) {
    out.innerHTML = "";
    out.appendChild(el("div", "err", `Error: ${e.message}`));
  }
}

// --------------------------------------------------------------------------
// Wire up
// --------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  $("#chat-form").addEventListener("submit", (e) => {
    e.preventDefault();
    sendMessage();
  });

  $("#build-btn").addEventListener("click", ingestYouTube);
  $("#clear-chat-btn").addEventListener("click", clearChatHistory);
  $("#clear-collection-btn").addEventListener("click", clearCollection);
  $("#clear-api-key-btn").addEventListener("click", clearApiKey);
  $("#time-bias").addEventListener("input", (e) => {
    $("#time-bias-value").textContent = parseFloat(e.target.value).toFixed(1);
  });
  $("#time-bias").addEventListener("change", (e) => setTimeBias(e.target.value));

  // Tabs
  document.querySelectorAll(".tab").forEach((t) => {
    t.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
      t.classList.add("active");
      const which = t.dataset.tab;
      $("#tab-cloud").classList.toggle("hidden", which !== "cloud");
      $("#tab-local").classList.toggle("hidden", which !== "local");
    });
  });

  $("#set-api-key-btn").addEventListener("click", setApiKey);
  $("#use-cloud-btn").addEventListener("click", () => {
    const name = $("#cloud-model").value.trim() || state.model;
    switchModel(name);
  });
  $("#use-local-btn").addEventListener("click", () => {
    const sel = $("#installed-select").value;
    const OTHER = "✏️ Other (type a name)…";
    let name = sel === OTHER ? $("#local-model").value.trim() : sel;
    if (!name) { alert("Enter a model name."); return; }
    switchModel(name);
  });
  $("#installed-select").addEventListener("change", (e) => {
    const OTHER = "✏️ Other (type a name)…";
    toggleLocalModelInput(e.target.value === OTHER);
  });

  // Chat mode radio
  document.querySelectorAll('input[name="chat-mode"]').forEach((r) => {
    r.addEventListener("change", (e) => {
      state.chatMode = e.target.value;
      // Refresh the global status chip immediately.
      const modeChip = $("#global-status").children[2];
      if (modeChip) modeChip.textContent = state.chatMode === "prose" ? "prose" : "lyric";
      applyChatInputMode();
    });
  });

  // Live-filter the chat input when in prose (single-letter) mode.
  $("#chat-input").addEventListener("input", (e) => {
    if (state.chatMode === "prose" && !(state.antakshari && state.antakshari.active)) {
      // Keep only the first alphabetic character.
      const m = (e.target.value.match(/[A-Za-z]/) || [""])[0];
      e.target.value = m;
    }
  });

  $("#antakshari-stop").addEventListener("click", stopAntakshari);

  fetchState();
});