const $ = (sel) => document.querySelector(sel);
const transcriptFeed = $("#transcriptFeed");
const answersFeed = $("#answersFeed");
const sessionBtn = $("#sessionBtn");
const calibrateBtn = $("#calibrateBtn");
const connectionDot = $("#connectionDot");
const questionCount = $("#questionCount");
const transcriptTimer = $("#transcriptTimer");
const summaryOverlay = $("#summaryOverlay");
const summaryContent = $("#summaryContent");
const closeSummary = $("#closeSummary");
const copySummary = $("#copySummary");

let ws = null;
let sessionActive = false;
let calibrated = false;
let calibrating = false;
let calibrateTimeout = null;
let timerInterval = null;
let sessionStartTime = null;
let qCount = 0;

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
      calibrated = msg.calibrated || false;
      updateSessionButton();
      updateCalibrateButton();
      if (msg.threads) {
        msg.threads.forEach((t) => {
          if (t.answer) renderAnswer(t);
        });
      }
      break;

    case "session_started":
      sessionActive = true;
      calibrated = msg.calibrated || false;
      sessionStartTime = Date.now();
      updateSessionButton();
      updateCalibrateButton();
      startTimer();
      clearFeeds();
      break;

    case "session_stopped":
      sessionActive = false;
      updateSessionButton();
      updateCalibrateButton();
      stopTimer();
      if (msg.summary) {
        showSummary(msg.summary);
      }
      break;

    case "calibrating":
      calibrating = true;
      calibrateBtn.textContent = "Speak now... (3s)";
      calibrateBtn.classList.add("calibrating");
      calibrateTimeout = setTimeout(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ action: "calibrate_stop" }));
        }
      }, 3000);
      break;

    case "calibrated":
      calibrating = false;
      calibrated = true;
      if (calibrateTimeout) { clearTimeout(calibrateTimeout); calibrateTimeout = null; }
      calibrateBtn.classList.remove("calibrating");
      updateCalibrateButton();
      break;

    case "calibrate_error":
      calibrating = false;
      if (calibrateTimeout) { clearTimeout(calibrateTimeout); calibrateTimeout = null; }
      calibrateBtn.classList.remove("calibrating");
      calibrateBtn.textContent = msg.message || "Calibration failed";
      setTimeout(updateCalibrateButton, 2000);
      break;

    case "transcript":
      appendTranscript(msg);
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

  }
}

function clearFeeds() {
  transcriptFeed.innerHTML = "";
  answersFeed.innerHTML = "";
  qCount = 0;
  questionCount.textContent = "0";
}

function appendTranscript(msg) {
  const empty = transcriptFeed.querySelector(".empty-state");
  if (empty) empty.remove();

  const line = document.createElement("div");
  line.className = "transcript-line";

  const speaker = msg.speaker || "Unknown";
  if (speaker === "You") {
    line.classList.add("speaker-self");
  } else if (speaker === "Other") {
    line.classList.add("speaker-other");
  }

  const mins = Math.floor(msg.start_time / 60);
  const secs = Math.floor(msg.start_time % 60);
  const ts = `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;

  const speakerTag = speaker !== "Unknown"
    ? `<span class="speaker-tag speaker-${speaker.toLowerCase()}">${speaker}</span>`
    : "";

  line.innerHTML = `<span class="timestamp">${ts}</span>${speakerTag}<span class="text">${escapeHtml(msg.text)}</span>`;
  transcriptFeed.appendChild(line);
  transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
}

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

function showSummary(markdown) {
  summaryContent.textContent = markdown;
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

function updateCalibrateButton() {
  calibrateBtn.disabled = !sessionActive;
  if (calibrated) {
    calibrateBtn.textContent = "Voice Calibrated";
    calibrateBtn.classList.add("calibrated");
  } else {
    calibrateBtn.textContent = "Calibrate Voice";
    calibrateBtn.classList.remove("calibrated");
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

calibrateBtn.addEventListener("click", () => {
  if (!ws || ws.readyState !== WebSocket.OPEN || !sessionActive || calibrating) return;
  ws.send(JSON.stringify({ action: "calibrate_start" }));
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

// Auto-connect on load
connect();
