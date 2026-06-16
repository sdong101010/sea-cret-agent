const $ = (sel) => document.querySelector(sel);
const transcriptFeed = $("#transcriptFeed");
const answersFeed = $("#answersFeed");
const sessionBtn = $("#sessionBtn");
const connectionDot = $("#connectionDot");
const questionCount = $("#questionCount");
const transcriptTimer = $("#transcriptTimer");
const summaryOverlay = $("#summaryOverlay");
const summaryContent = $("#summaryContent");
const closeSummary = $("#closeSummary");
const copySummary = $("#copySummary");
const createGoogleDocBtn = $("#createGoogleDoc");
const googleDocResult = $("#googleDocResult");

let ws = null;
let sessionActive = false;
let timerInterval = null;
let sessionStartTime = null;
let qCount = 0;
// When the user has opened a past meeting we render its contents into the
// main panels and remember which file is showing so "Create Google Doc"
// knows which markdown to send to Claude.
let viewingPastFile = null;
let viewingPastTitle = null;

function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws`);

  ws.onopen = () => {
    connectionDot.classList.add("connected");
  };

  ws.onclose = () => {
    connectionDot.classList.remove("connected");
    setTimeout(connect, 2000);
  };

  ws.onerror = () => {
    ws.close();
  };

  ws.onmessage = (evt) => {
    const msg = JSON.parse(evt.data);
    handleMessage(msg);
  };
}

function handleMessage(msg) {
  switch (msg.type) {
    case "status":
      sessionActive = msg.session_active;
      updateSessionButton();
      if (msg.threads) {
        msg.threads.forEach((t) => {
          if (t.answer) renderAnswer(t);
        });
      }
      break;

    case "session_started":
      sessionActive = true;
      sessionStartTime = Date.now();
      // Starting a fresh session yanks us out of any past-meeting view.
      if (viewingPastFile) {
        viewingPastFile = null;
        viewingPastTitle = null;
        if (pastBanner) pastBanner.hidden = true;
      }
      updateSessionButton();
      startTimer();
      clearFeeds();
      break;

    case "session_stopped":
      sessionActive = false;
      updateSessionButton();
      stopTimer();
      if (msg.summary) {
        showSummary(msg.summary);
      }
      break;

    case "transcript":
      appendTranscript(msg);
      break;

    case "transcript_update": {
      const line = document.querySelector(
        `.transcript-line[data-segment-id="${msg.id}"]`
      );
      if (!line) break;
      const textEl = line.querySelector(".text");
      if (textEl) textEl.textContent = msg.text;
      break;
    }

    case "speaker_renamed":
      applySpeakerRename(msg.old, msg.new);
      break;

    case "question":
      renderQuestionPending(msg);
      break;

    case "question_update":
      renderFollowUp(msg);
      break;

    case "answer":
      renderAnswer(msg);
      break;

    case "google_doc_started":
      createGoogleDocBtn.disabled = true;
      createGoogleDocBtn.textContent = "Creating...";
      googleDocResult.textContent = "Calling Google Docs (this can take 30-60s)...";
      googleDocResult.className = "google-doc-result pending";
      break;

    case "google_doc_result":
      createGoogleDocBtn.disabled = false;
      createGoogleDocBtn.textContent = "Create Google Doc";
      if (msg.error) {
        googleDocResult.textContent = "Failed: " + msg.error;
        googleDocResult.className = "google-doc-result error";
      } else {
        googleDocResult.innerHTML = `Created: <a href="${escapeHtml(msg.url)}" target="_blank" rel="noopener">${escapeHtml(msg.title || msg.url)}</a>`;
        googleDocResult.className = "google-doc-result success";
      }
      break;

  }
}

function clearFeeds() {
  transcriptFeed.innerHTML = "";
  answersFeed.innerHTML = "";
  qCount = 0;
  questionCount.textContent = "0";
}

function speakerCssClass(speaker) {
  // Stable, kebab-cased class so we can target all instances of the same
  // speaker for renames and color rotation.
  return "speaker-" + speaker.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

function appendTranscript(msg) {
  const empty = transcriptFeed.querySelector(".empty-state");
  if (empty) empty.remove();

  const line = document.createElement("div");
  line.className = "transcript-line";
  if (msg.id) line.dataset.segmentId = msg.id;

  const speaker = msg.speaker || "Unknown";
  if (speaker === "Me") {
    line.classList.add("speaker-self");
  } else if (speaker !== "Unknown") {
    line.classList.add("speaker-other");
  }

  const mins = Math.floor(msg.start_time / 60);
  const secs = Math.floor(msg.start_time % 60);
  const ts = `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;

  const isRemote = speaker !== "Unknown" && speaker !== "Me";
  const tagClasses = ["speaker-tag", speakerCssClass(speaker)];
  if (isRemote) tagClasses.push("renameable");
  const speakerTag = `<span class="${tagClasses.join(" ")}" data-speaker="${escapeHtml(speaker)}" ${isRemote ? 'title="Click to rename"' : ""}>${escapeHtml(speaker)}</span>`;

  line.innerHTML = `<span class="timestamp">${ts}</span>${speakerTag}<span class="text">${escapeHtml(msg.text)}</span>`;
  transcriptFeed.appendChild(line);
  transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
}

function applySpeakerRename(oldName, newName) {
  if (!oldName || !newName) return;
  const oldClass = speakerCssClass(oldName);
  const newClass = speakerCssClass(newName);
  document.querySelectorAll(`[data-speaker="${cssEscape(oldName)}"]`).forEach((el) => {
    el.dataset.speaker = newName;
    el.textContent = newName;
    el.classList.remove(oldClass);
    el.classList.add(newClass);
  });
}

function cssEscape(s) {
  // Minimal CSS attribute-selector escape for double quotes and backslashes.
  return String(s).replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

function promptRenameSpeaker(currentName) {
  const newName = window.prompt(`Rename "${currentName}" to:`, currentName);
  if (!newName) return;
  const trimmed = newName.trim();
  if (!trimmed || trimmed === currentName) return;
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: "rename_speaker", old: currentName, new: trimmed }));
  }
}

// Delegated click handler for renameable speaker tags.
transcriptFeed.addEventListener("click", (e) => {
  const tag = e.target.closest(".speaker-tag.renameable");
  if (!tag) return;
  const current = tag.dataset.speaker;
  if (current && current !== "Me" && current !== "Unknown") {
    promptRenameSpeaker(current);
  }
});

function renderQuestionPending(msg) {
  const empty = answersFeed.querySelector(".empty-state");
  if (empty) empty.remove();

  qCount++;
  questionCount.textContent = qCount;

  const card = document.createElement("div");
  card.className = "qa-card";
  card.id = `thread-${msg.thread_id}`;

  card.innerHTML = `
    <div class="qa-card-question">
      <div class="q-icon">Q</div>
      <div class="q-content">
        <p>${escapeHtml(msg.text)}</p>
        ${msg.raw_text !== msg.text ? `<p class="q-raw">Asked: "${escapeHtml(msg.raw_text)}"</p>` : ""}
      </div>
    </div>
    <div class="qa-card-meta">
      <span class="tag">${escapeHtml(msg.topic)}</span>
      <div class="confidence-bar">
        <div class="bar"><div class="bar-fill" style="width:${Math.round(msg.confidence * 100)}%"></div></div>
        ${Math.round(msg.confidence * 100)}%
      </div>
    </div>
    <div class="qa-card-answer searching">
      <p class="searching-indicator">Searching the web...</p>
    </div>
  `;

  answersFeed.prepend(card);
  answersFeed.scrollTop = 0;

  highlightTranscriptQuestion(msg.raw_text || msg.text);
}

function renderFollowUp(msg) {
  const card = document.querySelector(`#thread-${msg.thread_id}`);
  if (!card) {
    renderQuestionPending(msg);
    return;
  }

  const qContent = card.querySelector(".q-content") || card.querySelector(".qa-card-question");
  if (qContent) {
    qContent.innerHTML = `
      <p>${escapeHtml(msg.text)}</p>
      <p class="q-raw">Follow-up: "${escapeHtml(msg.raw_text)}"</p>
    `;
  }

  const meta = card.querySelector(".qa-card-meta");
  if (meta) {
    meta.innerHTML = `
      <span class="tag">${escapeHtml(msg.topic)}</span>
      <span class="tag tag-followup">Follow-up</span>
      <div class="confidence-bar">
        <div class="bar"><div class="bar-fill" style="width:${Math.round(msg.confidence * 100)}%"></div></div>
        ${Math.round(msg.confidence * 100)}%
      </div>
    `;
  }

  const answerDiv = card.querySelector(".qa-card-answer");
  if (answerDiv) {
    answerDiv.className = "qa-card-answer searching";
    answerDiv.innerHTML = `<p class="searching-indicator">Updating answer...</p>`;
  }

  highlightTranscriptQuestion(msg.raw_text || msg.text);
}

function renderAnswer(data) {
  const threadId = data.thread_id;
  let card = document.querySelector(`#thread-${threadId}`);

  if (!card) {
    const empty = answersFeed.querySelector(".empty-state");
    if (empty) empty.remove();

    qCount++;
    questionCount.textContent = qCount;

    card = document.createElement("div");
    card.className = "qa-card";
    card.id = `thread-${threadId}`;
    answersFeed.prepend(card);
  }

  const question = data.question || data.text || "";
  const answer = data.answer || "No answer available";
  const urls = data.urls || [];
  const topic = data.topic || "general";
  const confidence = data.confidence || 0;
  const suppressed = data.suppressed || false;

  if (suppressed) {
    card.classList.add("suppressed");
  } else {
    card.classList.remove("suppressed");
  }

  const linksHtml = urls.length
    ? `<div class="qa-card-links">${urls.map((link) => {
        const url = typeof link === "string" ? link : link.url;
        const label = typeof link === "string" ? prettyUrl(link) : (link.title || prettyUrl(url));
        return `<a href="${escapeHtml(url)}" target="_blank" rel="noopener" title="${escapeHtml(url)}">${escapeHtml(label)}</a>`;
      }).join("")}</div>`
    : "";

  const suppressedBadge = suppressed
    ? `<span class="tag tag-suppressed">Low confidence</span>`
    : "";

  card.innerHTML = `
    <div class="qa-card-question">
      <div class="q-icon">Q</div>
      <div class="q-content">
        <p>${escapeHtml(question)}</p>
      </div>
    </div>
    <div class="qa-card-meta">
      <span class="tag">${escapeHtml(topic)}</span>
      ${suppressedBadge}
      <div class="confidence-bar">
        <div class="bar"><div class="bar-fill" style="width:${Math.round(confidence * 100)}%"></div></div>
        ${Math.round(confidence * 100)}%
      </div>
    </div>
    <div class="qa-card-answer${suppressed ? " answer-suppressed" : ""}">
      ${suppressed ? `<p class="suppressed-msg">${escapeHtml(answer)}</p>` : formatAnswer(answer)}
    </div>
    ${suppressed ? "" : linksHtml}
  `;
}

function highlightTranscriptQuestion(questionText) {
  const lines = transcriptFeed.querySelectorAll(".transcript-line:not(.has-question)");
  const qWords = new Set(questionText.toLowerCase().split(/\s+/).filter(w => w.length > 3));

  for (let i = lines.length - 1; i >= Math.max(0, lines.length - 5); i--) {
    const lineText = lines[i].querySelector(".text")?.textContent?.toLowerCase() || "";
    const lineWords = new Set(lineText.split(/\s+/).filter(w => w.length > 3));
    const overlap = [...qWords].filter(w => lineWords.has(w)).length;
    if (overlap >= 3 || overlap / qWords.size > 0.5) {
      lines[i].classList.add("has-question");
    }
  }
}

function showSummary(markdown, opts = {}) {
  summaryContent.textContent = markdown;
  googleDocResult.textContent = "";
  googleDocResult.className = "google-doc-result";
  // Past sessions can now export to Google Docs too -- the server reads the
  // markdown from disk when we pass `filename` with the action.
  createGoogleDocBtn.style.display = "";
  createGoogleDocBtn.dataset.pastFile = opts.pastFile || "";
  summaryOverlay.classList.add("visible");
}

function updateSessionButton() {
  if (sessionActive) {
    sessionBtn.textContent = "Stop Session";
    sessionBtn.classList.add("active");
  } else {
    sessionBtn.textContent = "Start Session";
    sessionBtn.classList.remove("active");
  }
}

function startTimer() {
  sessionStartTime = Date.now();
  timerInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
    const m = Math.floor(elapsed / 60);
    const s = elapsed % 60;
    transcriptTimer.textContent = `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }, 1000);
}

function stopTimer() {
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function formatAnswer(text) {
  const lines = text.split("\n").map(l => l.trim()).filter(Boolean);
  const bullets = lines.filter(l => /^[•\-\*]/.test(l));
  if (bullets.length >= 2) {
    return "<ul>" + bullets.map(b =>
      `<li>${escapeHtml(b.replace(/^[•\-\*]\s*/, ""))}</li>`
    ).join("") + "</ul>";
  }
  return `<p>${escapeHtml(text)}</p>`;
}

function prettyUrl(url) {
  try {
    const u = new URL(url);
    const path = u.pathname.replace(/\/$/, "");
    const host = u.hostname.replace("www.", "");
    return path.length > 1 ? `${host}${path}` : host;
  } catch {
    return url;
  }
}

// --- Event listeners ---

sessionBtn.addEventListener("click", () => {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  if (sessionActive) {
    ws.send(JSON.stringify({ action: "stop" }));
  } else {
    ws.send(JSON.stringify({ action: "start" }));
  }
});

closeSummary.addEventListener("click", () => {
  summaryOverlay.classList.remove("visible");
});

copySummary.addEventListener("click", () => {
  navigator.clipboard.writeText(summaryContent.textContent).then(() => {
    copySummary.textContent = "Copied!";
    setTimeout(() => { copySummary.textContent = "Copy to Clipboard"; }, 2000);
  });
});

createGoogleDocBtn.addEventListener("click", () => {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const payload = { action: "create_google_doc" };
  const pastFile = createGoogleDocBtn.dataset.pastFile || "";
  if (pastFile) payload.filename = pastFile;
  ws.send(JSON.stringify(payload));
});

// --- Past meetings dropdown ---
const pastSessionsBtn = $("#pastSessionsBtn");
const pastSessionsMenu = $("#pastSessionsMenu");
const pastBanner = $("#pastBanner");
const pastBannerTitle = $("#pastBannerTitle");
const pastBannerSummaryBtn = $("#pastBannerSummary");
const pastBannerExitBtn = $("#pastBannerExit");

if (pastBannerSummaryBtn) pastBannerSummaryBtn.addEventListener("click", showPastSummary);
if (pastBannerExitBtn) pastBannerExitBtn.addEventListener("click", exitPastSession);

async function openPastSessionsMenu() {
  pastSessionsMenu.innerHTML = '<div class="past-sessions-loading">Loading…</div>';
  pastSessionsMenu.hidden = false;
  try {
    const res = await fetch("/api/summaries");
    const data = await res.json();
    const items = data.summaries || [];
    if (!items.length) {
      pastSessionsMenu.innerHTML = '<div class="past-sessions-empty">No past meetings yet.</div>';
      return;
    }
    pastSessionsMenu.innerHTML = items.map(it =>
      `<button class="past-session-item" data-file="${escapeHtml(it.file)}">
         <span class="past-session-when">${escapeHtml(it.when)}</span>
         <span class="past-session-title">${escapeHtml(it.title)}</span>
       </button>`
    ).join("");
    pastSessionsMenu.querySelectorAll(".past-session-item").forEach(btn => {
      btn.addEventListener("click", () => loadPastSession(btn.dataset.file));
    });
  } catch (e) {
    pastSessionsMenu.innerHTML = '<div class="past-sessions-empty">Couldn\'t load list.</div>';
  }
}

async function loadPastSession(filename) {
  pastSessionsMenu.hidden = true;
  try {
    const res = await fetch(`/api/summaries/${encodeURIComponent(filename)}/structured`);
    const data = await res.json();
    if (data.error || (!data.threads && !data.transcript)) {
      alert("Could not open: " + (data.error || "no content"));
      return;
    }
    renderPastSession(filename, data);
  } catch (e) {
    alert("Could not open past meeting.");
  }
}

function renderPastSession(filename, data) {
  viewingPastFile = filename;
  viewingPastTitle = data.title || "";

  // Reset the panels so the past meeting renders cleanly. We treat re-opens
  // like a fresh session view -- one source of truth on screen at a time.
  clearFeeds();
  pastBanner.hidden = false;
  pastBannerTitle.textContent = data.title || filename;
  sessionBtn.disabled = true;
  sessionBtn.title = "Exit the past meeting view to start a new session";

  if (data.transcript && data.transcript.length) {
    for (const seg of data.transcript) {
      appendTranscript(seg);
    }
  } else {
    transcriptFeed.innerHTML = '<div class="empty-state"><p>No transcript saved for this meeting.</p></div>';
  }

  if (data.threads && data.threads.length) {
    // renderAnswer prepends, so iterate in order to end up with Q1 on top --
    // matches the live session feel where the newest question is at the top.
    for (const t of [...data.threads].reverse()) {
      renderAnswer(t);
    }
  } else {
    answersFeed.innerHTML = '<div class="empty-state"><p>No questions detected in this meeting.</p></div>';
  }

  // Scroll transcript to the start so the user reads the meeting from the top.
  transcriptFeed.scrollTop = 0;
}

function exitPastSession() {
  viewingPastFile = null;
  viewingPastTitle = null;
  pastBanner.hidden = true;
  sessionBtn.disabled = false;
  sessionBtn.title = "";
  clearFeeds();
  transcriptFeed.innerHTML = '<div class="empty-state"><p>Waiting for audio...</p><p class="hint">Start a session, then speak or play meeting audio through BlackHole.</p></div>';
  answersFeed.innerHTML = '<div class="empty-state"><p>No questions detected yet</p><p class="hint">Questions from the customer will appear here with suggested answers.</p></div>';
}

async function showPastSummary() {
  if (!viewingPastFile) return;
  try {
    const res = await fetch(`/api/summaries/${encodeURIComponent(viewingPastFile)}`);
    const data = await res.json();
    if (data.markdown) {
      showSummary(data.markdown, { readOnly: false, pastFile: viewingPastFile });
    }
  } catch (e) {
    alert("Could not load summary.");
  }
}

pastSessionsBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  if (pastSessionsMenu.hidden) openPastSessionsMenu();
  else pastSessionsMenu.hidden = true;
});

document.addEventListener("click", (e) => {
  if (!pastSessionsMenu.hidden && !pastSessionsMenu.contains(e.target) && e.target !== pastSessionsBtn) {
    pastSessionsMenu.hidden = true;
  }
});

// Auto-connect on load
connect();
