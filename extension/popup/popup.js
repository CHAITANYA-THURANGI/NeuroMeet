const BACKEND_URL = "http://127.0.0.1:8000";

document.addEventListener("DOMContentLoaded", async () => {
  // Check backend health
  try {
    const res = await fetch(`${BACKEND_URL}/health`);
    if (res.ok) {
      document.getElementById("backendBadge").innerText = "● Online";
      document.getElementById("backendBadge").style.color = "#10b981";
      document.getElementById("backendBadge").style.borderColor = "rgba(16, 185, 129, 0.4)";
    }
  } catch (e) {
    document.getElementById("backendBadge").innerText = "● Offline";
    document.getElementById("backendBadge").style.color = "#f43f5e";
    document.getElementById("backendBadge").style.borderColor = "rgba(244, 63, 94, 0.4)";
  }

  // Query active tab
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const activeTab = tabs[0];
    if (activeTab && activeTab.url) {
      if (activeTab.url.includes("meet.google.com")) {
        document.getElementById("platformName").innerText = "Google Meet";
      } else if (activeTab.url.includes("zoom.us")) {
        document.getElementById("platformName").innerText = "Zoom Meeting";
      } else if (activeTab.url.includes("teams.microsoft.com")) {
        document.getElementById("platformName").innerText = "Microsoft Teams";
      } else {
        document.getElementById("platformName").innerText = "Web Browser Tab";
      }

      // Request captured turn count from content script
      chrome.tabs.sendMessage(activeTab.id, { action: "GET_CAPTURED_COUNT" }, (resp) => {
        if (chrome.runtime.lastError || !resp) {
          document.getElementById("turnCounter").innerText = "0";
        } else {
          document.getElementById("turnCounter").innerText = resp.count || 0;
        }
      });
    }
  });

  // Open Studio Dashboard
  document.getElementById("openStudioBtn").addEventListener("click", () => {
    chrome.tabs.create({ url: "http://127.0.0.1:8000" });
  });

  // Summarize Action
  document.getElementById("summarizeBtn").addEventListener("click", () => {
    const resBox = document.getElementById("quickResult");
    const preview = document.getElementById("summaryPreview");
    resBox.style.display = "block";
    preview.innerText = "Generating neural executive summary...";

    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      chrome.tabs.sendMessage(tabs[0].id, { action: "GET_TURNS" }, async (resp) => {
        if (!resp || !resp.turns || resp.turns.length === 0) {
          preview.innerText = "No captions captured yet. Please enable Closed Captions ('C' in Meet) to capture turns.";
          return;
        }

        try {
          const formatted = resp.turns.map(t => `${t.speaker}: ${t.text}`).join("\n");
          const apiRes = await fetch(`${BACKEND_URL}/api/v1/meetings/process-transcript`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title: "Live Meeting Session", transcript: formatted }),
          });
          const data = await apiRes.json();
          preview.innerHTML = `
            <strong>TL;DR:</strong> ${data.minutes.executive_summary}
            <div style="margin-top:6px; font-size:11px; color:#10b981;">
              <strong>Decisions (${data.minutes.key_decisions.length}):</strong>
              ${data.minutes.key_decisions.slice(0, 2).map(d => `<br>&bull; ${d}`).join("")}
            </div>
          `;
        } catch (err) {
          preview.innerText = "Error contacting NeuroMeet API at http://127.0.0.1:8000.";
        }
      });
    });
  });

  // Copy Action Items
  document.getElementById("copyActionsBtn").addEventListener("click", () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      chrome.tabs.sendMessage(tabs[0].id, { action: "GET_TURNS" }, async (resp) => {
        if (!resp || !resp.turns || resp.turns.length === 0) {
          alert("No speech turns recorded yet.");
          return;
        }
        try {
          const formatted = resp.turns.map(t => `${t.speaker}: ${t.text}`).join("\n");
          const apiRes = await fetch(`${BACKEND_URL}/api/v1/meetings/action-items`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ transcript: formatted }),
          });
          const data = await apiRes.json();
          if (data.action_items.length === 0) {
            alert("No commitments detected in the current discussion yet.");
            return;
          }
          const text = data.action_items.map(a => `- [ ] ${a.task} (Owner: ${a.assignee}, Due: ${a.deadline})`).join("\n");
          navigator.clipboard.writeText(text);
          alert(`Copied ${data.action_items.length} action items to clipboard!`);
        } catch (e) {
          alert("Error extracting tasks: " + e.message);
        }
      });
    });
  });
});
