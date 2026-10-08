// NeuroMeet Chrome Extension Content Script
// Captures live closed captions from Google Meet, Zoom, and Teams

(function () {
  let capturedTurns = [];
  let lastSpeaker = "";
  let lastText = "";

  console.log("[NeuroMeet] Live meeting assistant injected.");

  // Inject Floating HUD Pill
  const hud = document.createElement("div");
  hud.id = "neuromeet-hud";
  hud.innerHTML = `
    <div class="nm-hud-pill">
      <span class="nm-hud-icon">🧠</span>
      <span class="nm-hud-title">NeuroMeet</span>
      <span class="nm-hud-count" id="nm-hud-counter">0 turns</span>
      <button class="nm-hud-btn" id="nm-hud-action-btn">Summarize</button>
    </div>
    <div class="nm-hud-modal" id="nm-hud-modal" style="display:none;">
      <div class="nm-modal-header">
        <strong>NeuroMeet Live Insights</strong>
        <span class="nm-modal-close" id="nm-modal-close-btn">&times;</span>
      </div>
      <div class="nm-modal-content" id="nm-modal-content">Processing meeting turns...</div>
    </div>
  `;
  document.body.appendChild(hud);

  document.getElementById("nm-hud-action-btn").addEventListener("click", summarizeCurrentMeeting);
  document.getElementById("nm-modal-close-btn").addEventListener("click", () => {
    document.getElementById("nm-hud-modal").style.display = "none";
  });

  // Closed Captions Observer with Multi-Speaker Detection
  const observer = new MutationObserver(() => {
    // 1. Google Meet subtitle selector
    const meetCaptions = document.querySelectorAll('div[jsname="YSnbdc"], div.nMHgde, div.VbkSUe, span.yg7Jfc');
    meetCaptions.forEach((node) => {
      const text = node.innerText.trim();
      if (text && text !== lastText && text.length > 2) {
        lastText = text;
        const parent = node.closest('div[jsmodel="c6Tjhe"]') || node.closest('[data-sender-name]') || node.parentElement;
        const speakerEl = parent ? parent.querySelector('div.zs7LEd, div.NWxV7c, span.NWxV7c, div.KcIKyf') : null;
        let speaker = speakerEl ? speakerEl.innerText.trim() : (parent ? parent.getAttribute("data-sender-name") : null);
        if (!speaker) speaker = lastSpeaker || "Participant 1";
        lastSpeaker = speaker;

        capturedTurns.push({
          speaker: speaker,
          text: text,
          timestamp: Date.now(),
        });
        updateHudCounter();
      }
    });

    // 2. Microsoft Teams closed captions
    const teamsCaptions = document.querySelectorAll('div[data-tid="closed-caption-text"], .ui-chat__message__content');
    teamsCaptions.forEach((node) => {
      const text = node.innerText.trim();
      if (text && text !== lastText && text.length > 2) {
        lastText = text;
        const parent = node.closest('div[data-tid="closed-caption-item"]') || node.closest('.ui-chat__item') || node.parentElement;
        const speakerEl = parent ? parent.querySelector('span[data-tid="closed-caption-name"], .ui-chat__message__author') : null;
        const speaker = speakerEl ? speakerEl.innerText.trim() : (lastSpeaker || "Participant 1");
        lastSpeaker = speaker;

        capturedTurns.push({
          speaker: speaker,
          text: text,
          timestamp: Date.now(),
        });
        updateHudCounter();
      }
    });

    // 3. Zoom web captions
    const zoomCaptions = document.querySelectorAll('.caption-content, .subtitle-content, .meeting-client-caption');
    zoomCaptions.forEach((node) => {
      const text = node.innerText.trim();
      if (text && text !== lastText && text.length > 2) {
        lastText = text;
        const parent = node.closest('.caption-item') || node.parentElement;
        const speakerEl = parent ? parent.querySelector('.speaker-name, .name-tag, .meeting-client-speaker') : null;
        const speaker = speakerEl ? speakerEl.innerText.trim() : (lastSpeaker || "Participant 1");
        lastSpeaker = speaker;

        capturedTurns.push({
          speaker: speaker,
          text: text,
          timestamp: Date.now(),
        });
        updateHudCounter();
      }
    });
  });

  observer.observe(document.body, { childList: true, subtree: true, characterData: true });

  function updateHudCounter() {
    const el = document.getElementById("nm-hud-counter");
    if (el) el.innerText = `${capturedTurns.length} turns`;
  }

  async function summarizeCurrentMeeting() {
    const modal = document.getElementById("nm-hud-modal");
    const content = document.getElementById("nm-modal-content");
    modal.style.display = "block";

    if (capturedTurns.length === 0) {
      content.innerHTML = `<p style="color:#94a3b8;">No closed captions captured yet. Please ensure meeting captions are enabled (press 'C' in Google Meet).</p>`;
      return;
    }

    content.innerHTML = `<p style="color:#38bdf8;">Analyzing ${capturedTurns.length} speech turns via NeuroMeet Deep Learning...</p>`;

    try {
      const formatted = capturedTurns.map(t => `${t.speaker}: ${t.text}`).join("\n");
      const res = await fetch("http://127.0.0.1:8000/api/v1/meetings/process-transcript", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: "Live Meeting Capture", transcript: formatted }),
      });
      const data = await res.json();

      content.innerHTML = `
        <div style="margin-bottom:10px;">
          <strong style="color:#38bdf8;">Executive TL;DR:</strong>
          <p style="font-size:12px; margin-top:4px;">${data.minutes.executive_summary}</p>
        </div>
        <div style="margin-bottom:10px;">
          <strong style="color:#10b981;">Action Items (${data.action_items.length}):</strong>
          <ul style="font-size:12px; padding-left:16px; margin-top:4px;">
            ${data.action_items.map(a => `<li><strong>[${a.priority.toUpperCase()}]</strong> ${a.task} (Owner: ${a.assignee})</li>`).join("")}
          </ul>
        </div>
        <a href="http://127.0.0.1:8000" target="_blank" style="color:#818cf8; font-size:11px; text-decoration:underline;">Open Full Analytics Studio &rarr;</a>
      `;
    } catch (err) {
      content.innerHTML = `<p style="color:#ef4444;">Error connecting to NeuroMeet API at http://127.0.0.1:8000. Ensure local server is running.</p>`;
    }
  }

  // Chrome Extension Message Listener
  chrome.runtime.onMessage.addListener((req, sender, sendResponse) => {
    if (req.action === "GET_CAPTURED_COUNT") {
      sendResponse({ count: capturedTurns.length });
    } else if (req.action === "GET_TURNS") {
      sendResponse({ turns: capturedTurns });
    } else if (req.action === "TRIGGER_SUMMARIZE") {
      summarizeCurrentMeeting();
      sendResponse({ status: "ok" });
    }
  });
})();
